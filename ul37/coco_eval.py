#!/usr/bin/env python3
r"""Evaluate a released DEIMv2 COCO detector on COCO val2017 and score its detections.

Predict runs the repo's own config, val transforms and postprocessor in FP32 and writes a COCO results json: top-300
detections per image, no score threshold, 91-space COCO category ids (`remap_mscoco_category: True`) and xywh boxes in
original image pixels. Score evaluates that json with faster-coco-eval (bbox, default maxDets) against every given
annotation file, which must share the image ids and categories of the first one, and prints one SCORE line with AP,
AP50, APs, APm and APl as percentages.

`--engine` replaces the PyTorch forward with a static batch-1 TensorRT engine of the checkpoint, run with
`--batch 1`. Everything else stays, so its SCORE line, which then hashes the engine, compares directly with the
FP32 one.

Usage:
    CUDA_VISIBLE_DEVICES=0 python ul37/coco_eval.py configs/deimv2/deimv2_dinov3_x_coco.yml \
        model.safetensors --images coco/val2017 --name deimv2-x --output out \
        --ann coco=coco/annotations/instances_val2017.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo's engine package


def predict(
    config: Path,
    weights: Path,
    images: Path,
    ann_file: Path,
    output: Path,
    batch: int,
    engine: Path | None = None,
) -> None:
    """Write the COCO results json of a DEIMv2 checkpoint on every image of one annotation file.

    Args:
        config (Path): Repo config the checkpoint was trained with.
        weights (Path): Released safetensors state, or a checkpoint with `model` or `ema` state.
        images (Path): Directory holding the annotated images.
        ann_file (Path): COCO annotation json whose images are evaluated.
        output (Path): Results json to write.
        batch (int): Images per forward pass.
        engine (Path | None): Static batch-1 TensorRT engine of the checkpoint to run instead of PyTorch.
    """
    import torch
    from engine.core import YAMLConfig

    dataset = {"img_folder": str(images), "ann_file": str(ann_file)}
    cfg = YAMLConfig(
        str(config), remap_mscoco_category=True, val_dataloader={"dataset": dataset, "total_batch_size": batch}
    )
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False  # the checkpoint replaces the ImageNet backbone
    if weights.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(weights)
    else:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model, postprocessor = cfg.model.eval(), cfg.postprocessor.eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) <= {"decoder.reg_scale", "decoder.up"}, (missing, unexpected)  # constants
    model, postprocessor = model.cuda(), postprocessor.cuda()
    if engine:  # same loader and postprocessor around the engine, which carries an Ultralytics metadata header
        import tensorrt as trt

        data = engine.read_bytes()
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        context = runtime.deserialize_cuda_engine(
            data[4 + int.from_bytes(data[:4], "little") :]
        ).create_execution_context()
        io = {}
        for name in map(context.engine.get_tensor_name, range(context.engine.num_io_tensors)):
            dtype = getattr(torch, trt.nptype(context.engine.get_tensor_dtype(name)).__name__)
            io[name] = torch.empty(tuple(context.engine.get_tensor_shape(name)), dtype=dtype, device="cuda")
            context.set_tensor_address(name, io[name].data_ptr())

        def model(samples):
            io["images"].copy_(samples)
            context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            return io

    records = []
    with torch.no_grad():
        for samples, targets in cfg.val_dataloader:
            sizes = torch.stack([target["orig_size"] for target in targets]).cuda()
            for target, result in zip(targets, postprocessor(model(samples.cuda()), sizes)):
                for (x1, y1, x2, y2), score, label in zip(
                    result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist()
                ):
                    records.append(
                        {
                            "image_id": int(target["image_id"]),
                            "category_id": int(label),
                            "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                            "score": round(score, 5),
                        }
                    )
    output.write_text(json.dumps(records))


def score(results: Path, anns: dict[str, Path]) -> dict[str, list[float]]:
    """Return AP, AP50, APs, APm and APl in percent of a COCO results json against each annotation file.

    Args:
        results (Path): COCO results json.
        anns (dict[str, Path]): Annotation json by name.

    Returns:
        (dict[str, list[float]]): The five metrics by annotation name.
    """
    from faster_coco_eval import COCO, COCOeval_faster

    detections, scores, reference = json.loads(results.read_text()), {}, None
    for name, path in anns.items():
        gt = COCO(str(path))
        ids = (sorted(gt.getImgIds()), gt.getCatIds())
        assert reference in (None, ids), f"{path} differs from the first annotation file in image ids or categories"
        reference = ids
        evaluator = COCOeval_faster(gt, gt.loadRes(detections), iouType="bbox", print_function=lambda *a: None)
        evaluator.params.imgIds = ids[0]
        evaluator.evaluate(), evaluator.accumulate(), evaluator.summarize()
        stats = evaluator.stats_as_dict
        scores[name] = [round(stats[k] * 100, 2) for k in ("AP_all", "AP_50", "AP_small", "AP_medium", "AP_large")]
    return scores


def main() -> None:
    """Predict COCO val detections with one DEIMv2 checkpoint and score them."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path, help="repo config, e.g. configs/deimv2/deimv2_dinov3_s_coco.yml")
    parser.add_argument("weights", type=Path, help="released checkpoint")
    parser.add_argument("--images", type=Path, required=True, help="COCO val2017 image directory")
    parser.add_argument("--ann", nargs="+", required=True, help="NAME=PATH annotation jsons, the first drives predict")
    parser.add_argument("--output", type=Path, required=True, help="directory for the results json")
    parser.add_argument("--name", required=True, help="model name for the results file and SCORE line")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--engine", type=Path, help="TensorRT engine of the checkpoint to score, with --batch 1")
    args = parser.parse_args()

    anns = {name: Path(path) for name, path in (item.split("=", 1) for item in args.ann)}
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.output / f"{args.name}.json"
    predict(args.config, args.weights, args.images, next(iter(anns.values())), results, args.batch, args.engine)
    sha = hashlib.sha256((args.engine or args.weights).read_bytes()).hexdigest()[:8]
    print("SCORE", json.dumps({"model": args.name, "sha256": sha, **score(results, anns)}), flush=True)


if __name__ == "__main__":
    main()
