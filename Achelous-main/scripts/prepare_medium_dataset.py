#!/usr/bin/env python3
"""
Prepare a medium-scale aligned WaterScenes dataset with strict truncation.

Hard constraints implemented:
1) MAX_SAMPLES = 5000 (strictly from the first 5000 frames in train.txt)
2) All outputs are created under /home/waas/Project4/WaterScenes_Medium
3) Perfect alignment across image, radar npz, semantic, waterline, and 2007_train.txt

Usage:
    python scripts/prepare_medium_dataset.py
"""

import io
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

MAX_SAMPLES = 5000
ROOT = Path("/home/waas/Project4")
SRC = ROOT / "WaterScenes-Published"
OUT = ROOT / "WaterScenes_Medium"
RESOLUTION = 320

TRAIN_LIST = SRC / "train.txt"
IMAGE_ZIP = SRC / "image.zip"
DET_ZIP = SRC / "detection.zip"
SEM_ZIP = SRC / "semantic.zip"
WL_ZIP = SRC / "waterline.zip"
RADAR_ZIP = SRC / "radar.zip"

OUT_IMAGES = OUT / "images"
OUT_SEM = OUT / "semantic" / "SegmentationClass" / "SegmentationClass"
OUT_WL = OUT / "waterline" / "SegmentationClassPNG" / "SegmentationClassPNG"
OUT_RADAR = OUT / "radar" / "VOCradar320"
OUT_META = OUT / "meta"
OUT_FULL_TXT = OUT / "2007_train_full.txt"
OUT_TRAIN_TXT = OUT / "2007_train.txt"
OUT_VAL_TXT = OUT / "2007_val.txt"
PROJECT_DIR = ROOT / "Achelous-main"
PROJECT_TRAIN_TXT = PROJECT_DIR / "2007_train.txt"
PROJECT_VAL_TXT = PROJECT_DIR / "2007_val.txt"


def ensure_inputs_exist() -> None:
    required = [TRAIN_LIST, IMAGE_ZIP, DET_ZIP, SEM_ZIP, WL_ZIP, RADAR_ZIP]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n" + "\n".join(missing)
        )


def recreate_output_tree() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_SEM.mkdir(parents=True, exist_ok=True)
    OUT_WL.mkdir(parents=True, exist_ok=True)
    OUT_RADAR.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)


def frame_id_from_train_line(line: str) -> str:
    # train.txt line example: ./images/39484.jpg
    basename = os.path.basename(line.strip())
    return os.path.splitext(basename)[0]


def yolo_to_xyxy(yolo_text: str, img_w: int, img_h: int) -> list[str]:
    boxes = []
    for raw in yolo_text.splitlines():
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        x1 = max(0, int((cx - bw / 2.0) * img_w))
        y1 = max(0, int((cy - bh / 2.0) * img_h))
        x2 = min(img_w - 1, int((cx + bw / 2.0) * img_w))
        y2 = min(img_h - 1, int((cy + bh / 2.0) * img_h))

        if x2 > x1 and y2 > y1:
            boxes.append(f"{x1},{y1},{x2},{y2},{cls}")
    return boxes


def radar_csv_to_map(df: pd.DataFrame, resolution: int = RESOLUTION) -> np.ndarray:
    # Keep compatibility with project notebook logic.
    # Preferred features: range, doppler, rcs; fallback third channel to power.
    third = "rcs" if "rcs" in df.columns else "power"
    needed = ["range", "doppler", third, "u", "v"]
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Radar CSV missing column: {col}")

    points = df[needed].to_numpy()
    radar_map = np.zeros((3, resolution, resolution), dtype=np.float32)

    for c in range(3):
        for p in points:
            # Follow notebook: row from u, col from v
            row = int(float(p[-2]) / 6.0)
            col = int(float(p[-1]) / 3.375)
            if row < 0 or col < 0 or row >= resolution or col >= resolution:
                continue
            if radar_map[c, row, col] != 0 and row >= 1:
                row -= 1
            radar_map[c, row, col] = float(p[c])

    radar_map = radar_map.transpose(0, 2, 1)
    return radar_map


def main() -> None:
    ensure_inputs_exist()
    recreate_output_tree()

    with open(TRAIN_LIST, "r", encoding="utf-8") as f:
        train_lines = [ln.strip() for ln in f if ln.strip()]

    selected_lines = train_lines[:MAX_SAMPLES]
    if len(selected_lines) < MAX_SAMPLES:
        raise RuntimeError(
            f"train.txt only has {len(selected_lines)} lines, less than MAX_SAMPLES={MAX_SAMPLES}"
        )

    ids = [frame_id_from_train_line(ln) for ln in selected_lines]

    ann_lines: list[str] = []
    processed = 0

    with zipfile.ZipFile(IMAGE_ZIP) as zimg, \
         zipfile.ZipFile(DET_ZIP) as zdet, \
         zipfile.ZipFile(SEM_ZIP) as zsem, \
         zipfile.ZipFile(WL_ZIP) as zwl, \
         zipfile.ZipFile(RADAR_ZIP) as zrad:

        for frame_id in tqdm(ids, total=MAX_SAMPLES, desc="Preparing medium dataset"):
            if processed >= MAX_SAMPLES:
                break

            # Required paths in official zip structure.
            p_img = f"image/{frame_id}.jpg"
            p_det = f"detection/yolo/{frame_id}.txt"
            p_sem = f"semantic/SegmentationClass/{frame_id}.png"
            p_wl = f"waterline/SegmentationClass/{frame_id}.png"
            p_rad = f"radar/{frame_id}.csv"

            # Enforce complete alignment: if any modality missing, fail fast.
            required_members = [p_img, p_det, p_sem, p_wl, p_rad]
            for member in required_members:
                # zipfile.getinfo raises KeyError if missing.
                try:
                    if member.startswith("image/"):
                        zimg.getinfo(member)
                    elif member.startswith("detection/"):
                        zdet.getinfo(member)
                    elif member.startswith("semantic/"):
                        zsem.getinfo(member)
                    elif member.startswith("waterline/"):
                        zwl.getinfo(member)
                    else:
                        zrad.getinfo(member)
                except KeyError as e:
                    raise RuntimeError(f"Missing required member for frame {frame_id}: {member}") from e

            # Extract image and obtain size for bbox conversion.
            img_bytes = zimg.read(p_img)
            img_out = OUT_IMAGES / f"{frame_id}.jpg"
            img_out.write_bytes(img_bytes)
            with Image.open(io.BytesIO(img_bytes)) as im:
                img_w, img_h = im.size

            # Extract semantic + waterline.
            (OUT_SEM / f"{frame_id}.png").write_bytes(zsem.read(p_sem))
            (OUT_WL / f"{frame_id}.png").write_bytes(zwl.read(p_wl))

            # Convert radar csv -> npz feature map.
            radar_df = pd.read_csv(io.BytesIO(zrad.read(p_rad)))
            radar_map = radar_csv_to_map(radar_df, resolution=RESOLUTION)
            np.savez_compressed(OUT_RADAR / f"{frame_id}.npz", radar_map)

            # Convert YOLO txt -> 2007 style line.
            yolo_txt = zdet.read(p_det).decode("utf-8", errors="ignore")
            boxes = yolo_to_xyxy(yolo_txt, img_w, img_h)
            if boxes:
                ann_line = str(img_out) + " " + " ".join(boxes)
            else:
                # Keep parser compatibility if no bbox.
                ann_line = str(img_out) + " 0,0,1,1,0"
            ann_lines.append(ann_line)
            processed += 1

    # Strictly enforce exact sample count.
    if processed != MAX_SAMPLES:
        raise RuntimeError(f"Processed {processed}, expected exactly {MAX_SAMPLES}")

    # Save full 5000 lines first.
    OUT_FULL_TXT.write_text("\n".join(ann_lines) + "\n", encoding="utf-8")

    # 9:1 split -> 4500 train / 500 val (reproducible random split).
    rng = np.random.default_rng(20260419)
    perm = rng.permutation(MAX_SAMPLES)
    train_idx = perm[:4500]
    val_idx = perm[4500:5000]
    split_train = [ann_lines[int(i)] for i in train_idx]
    split_val = [ann_lines[int(i)] for i in val_idx]

    OUT_TRAIN_TXT.write_text("\n".join(split_train) + "\n", encoding="utf-8")
    OUT_VAL_TXT.write_text("\n".join(split_val) + "\n", encoding="utf-8")

    # Also write to project root so train.py can be launched directly.
    PROJECT_TRAIN_TXT.write_text("\n".join(split_train) + "\n", encoding="utf-8")
    PROJECT_VAL_TXT.write_text("\n".join(split_val) + "\n", encoding="utf-8")

    # Save alignment manifest for audit.
    (OUT_META / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    print("Done.")
    print(f"Output root: {OUT}")
    print(f"Samples: {processed}")
    print(f"2007_train_full: {OUT_FULL_TXT}")
    print(f"2007_train(4500): {OUT_TRAIN_TXT}")
    print(f"2007_val(500): {OUT_VAL_TXT}")
    print(f"Project train txt: {PROJECT_TRAIN_TXT}")
    print(f"Project val txt: {PROJECT_VAL_TXT}")
    print(f"Radar npz dir: {OUT_RADAR}")


if __name__ == "__main__":
    main()
