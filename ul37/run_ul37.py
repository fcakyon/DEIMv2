#!/usr/bin/env python3
"""Train D-FINE or DEIMv2 on Platform-backed UL37 datasets and save canonical prediction artifacts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]  # runs from <repo>/ul37
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ul37_common import (
    HEAD,
    PROTOCOL,
    TO_COCO,
    benchmark_json,
    check_init_load,
    grad_accum_steps,
    load_roster,
    lock_dataset,
    pending_datasets,
    run_benchmark,
    save_detr_predictions,
    sha256,
    write_args,
)

RUNNER_SHA256 = sha256(Path(__file__))  # at import: the code this process runs, even if redeployed under it
BATCH, SEED, IMGSZ, MAX_DET = 16, 0, 640, 300
TRAIN_WORKERS = 16  # throughput only, so outside the skip check
# Each family's custom-data recipe and its strongest published init: Objects365->COCO where released, else COCO.
FAMILIES = {
    "dfine": {
        "package": "src",
        "epochs_key": "epochs",
        "configs": {
            "n": "configs/dfine/custom/dfine_hgnetv2_n_custom.yml",
            **{s: f"configs/dfine/custom/objects365/dfine_hgnetv2_{s}_obj2custom.yml" for s in "smlx"},
        },
        "weights": {
            "n": "dfine/dfine_n_coco.pth",
            "s": "dfine/dfine_s_obj2coco.pth",
            "m": "dfine/dfine_m_obj2coco.pth",
            "l": "dfine/dfine_l_obj2coco_e25.pth",
            "x": "dfine/dfine_x_obj2coco.pth",
        },
        # obj2custom zeroes both warmups for an Objects365-mapped head; ours is fresh, so restore include defaults
        "warmups": {"ema": {"warmups": 1000}, "lr_warmup_scheduler": {"warmup_duration": 500}},
        "constants": (),
    },
    "deimv2": {
        "package": "engine",
        "epochs_key": "epoches",
        "configs": {
            "n": "configs/deimv2/deimv2_hgnetv2_n_coco.yml",
            **{s: f"configs/deimv2/deimv2_dinov3_{s}_coco.yml" for s in "smlx"},
        },
        "weights": {  # Hugging Face Intellindust/DEIMv2_<name>_COCO model.safetensors
            "n": "deimv2/HGNetv2_N/model.safetensors",
            **{s: f"deimv2/DINOv3_{s.upper()}/model.safetensors" for s in "smlx"},
        },
        "warmups": {},
        "constants": ("decoder.reg_scale", "decoder.up"),  # config constants the released weights omit
    },
}


def recipe(family: str, size: str, epochs: int) -> dict:
    """Return the UL37 overrides of the family's reference config, rescaled to the epochs and batch 16."""
    load_config = importlib.import_module(f"{FAMILIES[family]['package']}.core.yaml_utils").load_config
    reference = load_config(str(REPO / FAMILIES[family]["configs"][size]))
    key, train = FAMILIES[family]["epochs_key"], reference["train_dataloader"]
    lr_scale = BATCH / train["total_batch_size"]  # linear LR scaling from the config's global batch

    def epoch(value: int | list[int]) -> int | list[int]:
        return [epoch(v) for v in value] if isinstance(value, list) else round(value * epochs / reference[key])

    collate = train["collate_fn"]
    return {
        key: epochs,
        "checkpoint_freq": epochs,  # no periodic checkpoint archive
        "optimizer": {
            "lr": reference["optimizer"]["lr"] * lr_scale,
            "params": [
                {**group, "lr": group["lr"] * lr_scale} if "lr" in group else group
                for group in reference["optimizer"]["params"]
            ],
        },
        "train_dataloader": {
            "total_batch_size": BATCH,
            "dataset": {"transforms": {"policy": {"epoch": epoch(train["dataset"]["transforms"]["policy"]["epoch"])}}},
            "collate_fn": {
                name: epoch(collate[name])
                for name in ("stop_epoch", "mixup_epochs", "copyblend_epochs")
                if name in collate
            },
        },
        **{name: epoch(reference[name]) for name in ("flat_epoch", "no_aug_epoch") if name in reference},
        **{
            criterion: {"matcher": {"matcher_change_epoch": epoch(matcher["matcher_change_epoch"])}}
            for criterion in [reference["criterion"]]
            if "matcher_change_epoch" in (matcher := reference[criterion].get("matcher", {}))
        },
        **({"HGNetv2": {"pretrained": False}} if "HGNetv2" in reference else {}),  # the init replaces ImageNet weights
        **FAMILIES[family]["warmups"],
        "lr_scale": lr_scale,
    }


def build_init(family: str, size: str, weights: Path) -> Path:
    """Save the released detector without its class-shaped head tensors, so every dataset gets a fresh head."""
    import torch

    source = weights / FAMILIES[family]["weights"][size]
    init = REPO / f"weights/ul37/{family}_{size}_ul37_init.pth"
    if not init.is_file():
        if source.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(source)
        else:
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        init.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {k: v for k, v in state.items() if not k.startswith(HEAD)}}, init)
    return init


def save_predictions(family: str, config: Path, checkpoint: Path, output: Path) -> tuple[int, int, int]:
    """Save the final checkpoint's EMA detections of the val split in the canonical Ultra Benchmark JSON shape."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():  # model builders query the rank
        dist.init_process_group("gloo", init_method=f"file://{output.parent}/.pg", rank=0, world_size=1)
    YAMLConfig = importlib.import_module(f"{FAMILIES[family]['package']}.core").YAMLConfig
    cfg = YAMLConfig(str(config))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, postprocessor = cfg.model.cuda().eval(), cfg.postprocessor.cuda().eval()
    model.load_state_dict(state["ema"]["module"], strict=True)
    ann_file = Path(cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"])
    val_images = save_detr_predictions(model, postprocessor, cfg.val_dataloader, ann_file, output)
    torch.save({"model": state["ema"]["module"], "last_epoch": state["last_epoch"]}, output.with_name("final_ema.pth"))
    dist.destroy_process_group()
    return state["ema"]["module"]["decoder.enc_score_head.bias"].numel(), val_images, state["last_epoch"]


def train_run(args: argparse.Namespace, run_dir: Path) -> None:
    """Train every selected dataset without args.yaml, then predict from its final checkpoint."""
    fraction, datasets, data_yamls, ultralytics_sha = load_roster(args.benchmark)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import torch

    family = FAMILIES[args.family]
    overrides = recipe(args.family, args.size, args.epochs)
    lr_scale = overrides.pop("lr_scale")
    source = args.weights / family["weights"][args.size]
    common = {
        "family": args.family,
        "size": args.size,
        "repo_sha": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "repo_patch_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(REPO), "diff", "HEAD"])
        ).hexdigest(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "ultralytics_sha": ultralytics_sha,
        "roster_sha256": sha256(args.benchmark / "ul37_platform_uris.txt"),
        "fraction": fraction,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "seed": SEED,
        "amp": True,
        "lr_scale": lr_scale,
        "source_sha256": sha256(source),
        **overrides,
    }
    pending = pending_datasets(run_dir, datasets, args.subset, common)
    init = build_init(args.family, args.size, args.weights) if pending else None
    for name, expected in pending.items():
        output_dir, scratch = run_dir / name, Path.home() / f"{args.family}-scratch" / run_dir.parent.name / name
        lock = lock_dataset(run_dir, name)  # noqa: F841, held until the next dataset
        shutil.rmtree(output_dir, ignore_errors=True)  # no args.yaml, so anything here is from an unfinished attempt
        output_dir.mkdir(parents=True)
        (scratch / "train").mkdir(parents=True, exist_ok=True)
        num_classes = benchmark_json(args.benchmark, TO_COCO, str(data_yamls[name]), str(scratch))
        config = {
            "__include__": [str(REPO / family["configs"][args.size])],
            "num_classes": num_classes,
            "remap_mscoco_category": False,
            "output_dir": str(scratch / "train"),
            **yaml.safe_load(yaml.safe_dump(overrides)),
        }
        for split in ("train", "val"):
            config.setdefault(f"{split}_dataloader", {}).setdefault("dataset", {}).update(
                img_folder="/", ann_file=str(scratch / f"{split}.json")
            )
        config["train_dataloader"]["num_workers"] = TRAIN_WORKERS
        # The fork's GRAD_ACCUM_STEPS keeps 16 images per optimizer step. An OOM halves the micro-batch and resumes.
        # last.pth stops updating at the no-aug stage, so a crash there restarts that stage from best_stg1.pth.
        accum, last, log_path = (
            grad_accum_steps(scratch, BATCH),
            scratch / "train/last.pth",
            scratch / "train/train.log",
        )
        while True:
            config["train_dataloader"]["total_batch_size"] = BATCH // accum
            (scratch / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
            print(
                f"{name}: {data_yamls[name].parent} num_classes={num_classes} grad_accum={accum} "
                f"resume={last.is_file()}",
                flush=True,
            )
            start = log_path.stat().st_size if log_path.is_file() else 0
            with log_path.open("a") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node=1",
                        "train.py",
                        "-c",
                        str(scratch / "config.yml"),
                        "--use-amp",
                        f"--seed={SEED}",
                        *(["-r", str(last)] if last.is_file() else ["-t", str(init)]),
                    ],
                    cwd=REPO,
                    check=False,
                    env={**os.environ, "GRAD_ACCUM_STEPS": str(accum)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if (
                result.returncode == 0
                or accum == BATCH
                or b"torch.OutOfMemoryError" not in log_path.read_bytes()[start:]
            ):
                result.check_returncode()
                break
            accum *= 2
            (scratch / "grad_accum").write_text(str(accum))
        check_init_load(log_path.read_text(), family["constants"])
        final = next(
            path for path in (scratch / "train/best_stg2.pth", scratch / "train/best_stg1.pth") if path.is_file()
        )
        head, val_images, final_epoch = save_predictions(
            args.family, scratch / "config.yml", final, output_dir / "predictions.json"
        )
        assert head == num_classes, (head, num_classes)
        gc.collect()
        torch.cuda.empty_cache()
        for path in [scratch / "config.yml", *(scratch / "train").iterdir()]:  # logs and config, no checkpoints
            if path.suffix != ".pth" and path.name != "eval":
                shutil.move(path, output_dir / path.name)
        shutil.rmtree(scratch)
        (output_dir / "final_ema.pth").chmod(0o664)
        write_args(
            output_dir,
            {
                **expected,
                "runner_sha256": RUNNER_SHA256,
                "data": str(data_yamls[name]),
                "num_classes": num_classes,
                "max_det": MAX_DET,
                "micro_batch": BATCH // accum,
                "grad_accum_steps": accum,
                "num_workers": TRAIN_WORKERS,
                "source_model": str(source),
                "init_sha256": sha256(init),
                "final_checkpoint": final.name,
                "final_epoch": final_epoch,
                "val_images": val_images,
                "prediction_checkpoint": str(output_dir / "final_ema.pth"),
            },
        )


def main() -> None:
    """Train or score one D-FINE or DEIMv2 UL37 run."""
    os.umask(0o002)
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=tuple(FAMILIES))
    parser.add_argument("size", choices=tuple("nsmlx"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--subset", help="Dataset slice like 0:5 or comma-separated dataset names")
    parser.add_argument("--name")
    parser.add_argument("--project", type=Path, required=True, help="Output root, runs go to PROJECT/NAME/multitrain")
    parser.add_argument("--benchmark", type=Path, required=True, help="UL37 benchmark checkout with val_ultrabench.py")
    parser.add_argument("--weights", type=Path, required=True, help="Released weights root holding dfine/ and deimv2/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--native-size-bins", action="store_true")
    args = parser.parse_args()

    name = args.name or f"ultrabench-{PROTOCOL}-{args.family}-{args.size}"
    run_dir = (args.project / name / "multitrain").resolve()
    if not args.val_only:
        train_run(args, run_dir)
    if args.val_only or not args.subset:
        run_benchmark(
            args.benchmark,
            "val_ultrabench.py",
            str(run_dir),
            *(["--native-size-bins"] if args.native_size_bins else []),
        )


if __name__ == "__main__":
    main()
