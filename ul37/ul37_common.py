"""Shared UL37 runner helpers: benchmark runtime, roster, dataset conversion, DETR init checks and predictions."""

from __future__ import annotations

import ast
import collections
import fcntl
import functools
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

for library in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):  # before any runner imports numpy or torch
    os.environ.setdefault(f"{library}_NUM_THREADS", "1")

PROTOCOL = "081026"
BOXES_PER_MICRO_BATCH = 2048  # micro-batch times max GT boxes per train image, calibrated on GTR-X
HEAD = ("decoder.enc_score_head", "decoder.dec_score_head", "decoder.denoising_class_embed")  # DEIM class heads
TO_COCO = """
import json, sys
from pathlib import Path
from PIL import Image
from ultralytics.cfg import get_cfg
import ultralytics.data.utils as data_utils
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
data_utils.is_dir_writeable = lambda path: False  # label caches are read, never written into shared datasets
data = check_det_dataset(sys.argv[1], autodownload=False)
cfg = get_cfg(overrides={"task": "detect", "imgsz": 640, "cache": False, "workers": 0})
categories = [{"id": int(i), "name": name} for i, name in data["names"].items()]
for split in ("train", "val"):
    dataset = build_yolo_dataset(cfg, data[split], 16, data, mode="val", stride=32)
    images, annotations = [], []
    for image_id, (path, label) in enumerate(zip(dataset.im_files, dataset.labels)):
        height, width = label["shape"]
        with Image.open(path) as image:  # DETR loaders read with PIL and never apply EXIF orientation
            assert image.size == (width, height) and image.getexif().get(0x0112, 1) == 1, path
        images.append({"id": image_id, "file_name": path, "height": height, "width": width})
        for cls, (x, y, w, h) in zip(label["cls"].reshape(-1).tolist(), label["bboxes"].tolist()):
            box = [(x - w / 2) * width, (y - h / 2) * height, w * width, h * height]
            annotations.append(
                {"id": len(annotations) + 1, "image_id": image_id, "category_id": int(cls), "bbox": box,
                 "area": box[2] * box[3], "iscrowd": 0}
            )
    Path(sys.argv[2], f"{split}.json").write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
print(json.dumps(len(categories)))
"""


def run_benchmark(benchmark: Path, *args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run the benchmark checkout's isolated UL37 Python, .venv-ul37 with .config-ul37-<protocol>."""
    env = {**os.environ, "YOLO_CONFIG_DIR": str(benchmark / f".config-ul37-{PROTOCOL}"), "YOLO_AUTOINSTALL": "False"}
    return subprocess.run(
        [str(benchmark / ".venv-ul37/bin/python"), *args], cwd=benchmark, check=True, text=True, env=env, **kwargs
    )


def lock_dataset(run_dir: Path, name: str):
    """Hold a dataset's lock in the shared run directory, so a second runner of it on any host fails at once."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / f".{name}.lock").open("w")
    fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # POSIX locks, unlike flock, also bind the NFS server's own host
    return lock


def benchmark_json(benchmark: Path, code: str, *args: str):
    """Return the JSON that Python code prints last in the benchmark runtime."""
    return json.loads(run_benchmark(benchmark, "-c", code, *args, stdout=subprocess.PIPE).stdout.splitlines()[-1])


def load_roster(benchmark: Path) -> tuple[list, dict[str, str], dict[str, Path], str]:
    """Return the benchmark fraction, Platform URIs and pinned data.yaml paths by dataset, and the Ultralytics sha."""
    code = (
        "import json, importlib.metadata as m; from val_ultrabench import FRACTION, load_datasets, load_dataset_yamls; "
        "sha = json.loads(m.distribution('ultralytics').read_text('direct_url.json'))['vcs_info']['commit_id']; "
        "print(json.dumps([FRACTION, load_datasets(), {k: str(v) for k, v in load_dataset_yamls().items()}, sha]))"
    )
    fraction, datasets, data_yamls, sha = benchmark_json(benchmark, code)
    return fraction, datasets, {name: Path(path).resolve() for name, path in data_yamls.items()}, sha


def select_datasets(names: list[str], subset: str) -> list[str]:
    """Select a START:STOP roster slice or comma-separated dataset names in canonical order."""
    if ":" in subset:
        selected = names[slice(*(int(value) if value else None for value in subset.split(":")))]
    else:
        requested = subset.split(",")
        if unknown := set(requested) - set(names):
            raise ValueError(f"Unknown UL37 datasets: {sorted(unknown)}")
        selected = [name for name in names if name in requested]
    if not selected:
        raise ValueError(f"Subset {subset!r} selects no datasets")
    return selected


def grad_accum_steps(scratch: Path, batch: int) -> int:
    """Return saved accumulation steps, else the fewest that keep micro-batch times max image boxes in budget.

    Denoising queries grow with the box count of a batch's densest image, so memory scales with micro-batch times that
    count. The GTR-X calibrated BOXES_PER_MICRO_BATCH is conservative for D-FINE and DEIMv2.
    """
    accum_file = scratch / "grad_accum"
    if accum_file.is_file():
        return int(accum_file.read_text())
    annotations = json.loads((scratch / "train.json").read_text())["annotations"]
    max_boxes = max(collections.Counter(a["image_id"] for a in annotations).values())
    return batch // next(micro for micro in (16, 8, 4, 2, 1) if micro * max_boxes <= BOXES_PER_MICRO_BATCH)


def pending_datasets(run_dir: Path, datasets: dict[str, str], subset: str | None, common: dict) -> dict[str, dict]:
    """Return each selected dataset without args.yaml and its expected provenance, refusing changed finished ones."""
    pending = {}
    for name in select_datasets(list(datasets), subset) if subset else datasets:
        expected = {"platform_uri": datasets[name], **common}
        args_path = run_dir / name / "args.yaml"
        if not args_path.is_file():
            pending[name] = expected
            continue
        saved = yaml.safe_load(args_path.read_text())
        if mismatched := {key: (saved.get(key), value) for key, value in expected.items() if saved.get(key) != value}:
            raise ValueError(f"Refusing to skip {name} with changed provenance: {mismatched}")
        print(f"{name}: complete, skipping")
    return pending


def check_init_load(log: str, constants: tuple[str, ...] = ()) -> None:
    """Assert a DEIM-family tuning load missed only the class head and known config constants."""
    line = next(line for line in log.splitlines() if line.startswith("Load model.state_dict, {"))
    infos = ast.literal_eval(line.split(", ", 1)[1])
    assert not infos["unmatched"] and all(k.startswith(HEAD + constants) for k in infos["missed"]), infos
    assert any(k.startswith(HEAD) for k in infos["missed"]), infos


def save_detr_predictions(model, postprocessor, loader, ann_file: Path, output: Path) -> int:
    """Write DEIM-family val detections in the canonical Ultra Benchmark JSON shape and return the val image count."""
    import torch

    files = {image["id"]: Path(image["file_name"]).name for image in json.loads(ann_file.read_text())["images"]}
    records = []
    with torch.no_grad():
        for samples, targets in loader:
            sizes = torch.stack([target["orig_size"] for target in targets]).cuda()
            for target, result in zip(targets, postprocessor(model(samples.cuda()), sizes)):
                name = files[int(target["image_id"])]
                for (x1, y1, x2, y2), score, label in zip(
                    result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist()
                ):
                    records.append(
                        {
                            "file_name": name,
                            "category_id": int(label) + 1,
                            "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                            "score": round(score, 5),
                        }
                    )
    output.write_text(json.dumps(records))
    return len(files)


def write_args(output_dir: Path, provenance: dict) -> None:
    """Write args.yaml last and atomically, since it marks a finished dataset."""
    temporary = output_dir / "args.tmp"
    temporary.write_text(yaml.safe_dump(provenance, sort_keys=False))
    temporary.replace(output_dir / "args.yaml")


@functools.lru_cache
def sha256(path: Path) -> str:
    """Calculate one file SHA-256 digest, cached because weights repeat across datasets."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
