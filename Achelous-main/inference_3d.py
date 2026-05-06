#!/usr/bin/env python3
"""
Achelous 2D->3D inference prototype.

What this script does:
1) Runs Achelous3T inference on one image + one radar npz map.
2) Gets 2D boxes [x1, y1, x2, y2].
3) Extracts range/depth Z from radar npz region (median by default).
4) Converts pixel center to 3D point (X, Y, Z) with pinhole model.
5) Estimates real-world width/height from box size.

This is a framework script for graduation-project 3D highlight code.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import boxes

from nets.Achelous import Achelous3T
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import yolo_correct_boxes


@dataclass
class Detection3D:
    class_id: int
    class_name: str
    score: float
    box_xyxy: List[float]
    pixel_center: List[float]
    depth_z: float
    xyz: List[float]
    est_width_m: float
    est_height_m: float


def decode_outputs_local(outputs: List[torch.Tensor], input_shape: List[int]) -> torch.Tensor:
    """Decode Achelous/YOLOX-style outputs without forcing CUDA device."""
    hw = [x.shape[-2:] for x in outputs]
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])

    grids = []
    strides = []
    for h, w in hw:
        gy, gx = torch.meshgrid(torch.arange(h, device=outputs.device), torch.arange(w, device=outputs.device), indexing="ij")
        grid = torch.stack((gx, gy), 2).view(1, -1, 2)
        shape = grid.shape[:2]
        stride = torch.full((shape[0], shape[1], 1), input_shape[0] / h, device=outputs.device)
        grids.append(grid)
        strides.append(stride)

    grids = torch.cat(grids, dim=1).type(outputs.dtype)
    strides = torch.cat(strides, dim=1).type(outputs.dtype)

    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
    outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]
    return outputs


def non_max_suppression_local(
    prediction: torch.Tensor,
    num_classes: int,
    input_shape: List[int],
    image_shape: np.ndarray,
    letterbox_image: bool,
    conf_thres: float,
    nms_thres: float,
) -> Optional[np.ndarray]:
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    image_pred = prediction[0]
    class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
    conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
    detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
    detections = detections[conf_mask]
    if detections.size(0) == 0:
        return None

    keep = boxes.batched_nms(
        detections[:, :4],
        detections[:, 4] * detections[:, 5],
        detections[:, 6],
        nms_thres,
    )
    detections = detections[keep].detach().cpu().numpy()
    box_xy = (detections[:, 0:2] + detections[:, 2:4]) / 2
    box_wh = detections[:, 2:4] - detections[:, 0:2]
    detections[:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return detections


def select_depth_from_radar_region(
    radar_map: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    use_median: bool = True,
) -> float:
    """Use channel-0 as range depth; robustly aggregate non-zero values in bbox region."""
    range_map = radar_map[0]
    h, w = range_map.shape
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return float("nan")

    region = range_map[y1:y2, x1:x2]
    valid = region[region > 0]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid) if use_median else np.mean(valid))


def pinhole_to_3d(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def load_intrinsic_from_calib(calib_txt: Path) -> tuple[np.ndarray, float, float, float, float]:
    """Parse line-2 t_camera_intrinsic from calib txt as 3x4 matrix and derive fx/fy/cx/cy."""
    if not calib_txt.exists():
        raise FileNotFoundError(f"Calib txt not found: {calib_txt}")

    lines = calib_txt.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid calib format (need >=2 lines): {calib_txt}")

    line2 = lines[1].strip()
    if ":" not in line2:
        raise ValueError(f"Invalid intrinsic line: {line2}")

    _, values_text = line2.split(":", 1)
    vals = [float(v) for v in values_text.strip().split()]
    if len(vals) != 12:
        raise ValueError(f"Intrinsic expects 12 values (3x4), got {len(vals)} in {calib_txt}")

    k34 = np.array(vals, dtype=np.float32).reshape(3, 4)
    fx = float(k34[0, 0])
    fy = float(k34[1, 1])
    cx = float(k34[0, 2])
    cy = float(k34[1, 2])
    return k34, fx, fy, cx, cy


def build_model(weights: Path, classes_path: Path, input_shape: List[int], device: torch.device) -> tuple[nn.Module, List[str]]:
    class_names, num_classes = get_classes(str(classes_path))
    model = Achelous3T(
        resolution=input_shape[0],
        num_det=num_classes,
        num_seg=9,
        phi="S0",
        backbone="mo",
        neck="rdf",
        nano_head=True,
        spp=True,
    )
    state = torch.load(str(weights), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, class_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Achelous 3D inference prototype")
    parser.add_argument("--weights", type=str, default="/home/waas/Project4/Achelous-main/logs/best_epoch_weights.pth")
    parser.add_argument("--classes", type=str, default="/home/waas/Project4/Achelous-main/model_data/waterscenes_benchmark.txt")
    parser.add_argument("--image", type=str, required=True, help="Path to one test image")
    parser.add_argument("--radar_npz", type=str, required=True, help="Path to corresponding radar npz")
    parser.add_argument("--input_size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--nms_iou", type=float, default=0.35)
    parser.add_argument("--data_root", type=str, default="/home/waas/Project4/WaterScenes_Medium")
    parser.add_argument("--calib_txt", type=str, default="", help="Optional explicit calib txt path")
    parser.add_argument("--target_class", type=str, default="", help="Optional class name filter")
    parser.add_argument("--use_median", action="store_true", default=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    image_path = Path(args.image)
    radar_npz_path = Path(args.radar_npz)
    weights_path = Path(args.weights)
    classes_path = Path(args.classes)
    data_root = Path(args.data_root)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not radar_npz_path.exists():
        raise FileNotFoundError(f"Radar npz not found: {radar_npz_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if args.calib_txt:
        calib_txt_path = Path(args.calib_txt)
    else:
        calib_txt_path = data_root / "calib" / f"{image_path.stem}.txt"

    k34, fx, fy, cx, cy = load_intrinsic_from_calib(calib_txt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = [args.input_size, args.input_size]

    model, class_names = build_model(weights_path, classes_path, input_shape, device)

    image = Image.open(image_path)
    image = cvtColor(image)
    image_shape = np.array(np.shape(image)[0:2])
    resized = resize_image(image, (input_shape[1], input_shape[0]), True)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(resized, dtype="float32")), (2, 0, 1)), 0)

    radar_arr = np.load(radar_npz_path)["arr_0"].astype(np.float32)
    radar_tensor = torch.from_numpy(radar_arr).unsqueeze(0).to(device)
    image_tensor = torch.from_numpy(image_data).to(device)

    with torch.no_grad():
        outputs, _, _ = model(image_tensor, radar_tensor)
        decoded = decode_outputs_local(outputs, input_shape)
        dets = non_max_suppression_local(
            decoded,
            num_classes=len(class_names),
            input_shape=input_shape,
            image_shape=image_shape,
            letterbox_image=True,
            conf_thres=args.confidence,
            nms_thres=args.nms_iou,
        )

    results: List[Detection3D] = []
    if dets is not None:
        for row in dets:
            x1, y1, x2, y2, obj_conf, cls_conf, cls_id = row
            cls_id_i = int(cls_id)
            cls_name = class_names[cls_id_i]
            score = float(obj_conf * cls_conf)

            if args.target_class and cls_name != args.target_class:
                continue

            bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
            u = (bx1 + bx2) / 2.0
            v = (by1 + by2) / 2.0
            z = select_depth_from_radar_region(radar_arr, bx1, by1, bx2, by2, use_median=args.use_median)

            if np.isnan(z) or np.isinf(z):
                xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
                width_m = float("nan")
                height_m = float("nan")
            else:
                xyz = pinhole_to_3d(u, v, z, fx, fy, cx, cy)
                width_px = max(1.0, float(bx2 - bx1))
                height_px = max(1.0, float(by2 - by1))
                width_m = width_px * z / fx
                height_m = height_px * z / fy

            results.append(
                Detection3D(
                    class_id=cls_id_i,
                    class_name=cls_name,
                    score=score,
                    box_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                    pixel_center=[float(u), float(v)],
                    depth_z=float(z),
                    xyz=[float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    est_width_m=float(width_m),
                    est_height_m=float(height_m),
                )
            )

    payload = {
        "image": str(image_path),
        "radar_npz": str(radar_npz_path),
        "calib_txt": str(calib_txt_path),
        "camera_intrinsic_3x4": k34.tolist(),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "num_detections": len(results),
        "detections_3d": [asdict(r) for r in results],
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
