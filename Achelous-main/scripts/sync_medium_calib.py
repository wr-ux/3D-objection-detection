#!/usr/bin/env python3
"""Sync calibration txt files for WaterScenes_Medium exact frame list.

Reads frame ids from 2007_train_full.txt (first token of each line), then extracts
matching calib/<id>.txt from WaterScenes-Published/calib.zip into
WaterScenes_Medium/calib/ with strict one-to-one validation.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile


def load_ids(list_path: Path) -> list[str]:
    ids: list[str] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        img_path = line.split()[0]
        ids.append(Path(img_path).stem)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync calib txt files for Medium dataset")
    parser.add_argument("--list", required=True, help="Path to 2007_train_full.txt")
    parser.add_argument("--zip", required=True, help="Path to calib.zip")
    parser.add_argument("--out", required=True, help="Output calib directory")
    args = parser.parse_args()

    list_path = Path(args.list)
    zip_path = Path(args.zip)
    out_dir = Path(args.out)

    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    frame_ids = load_ids(list_path)
    if len(frame_ids) == 0:
        raise RuntimeError("No frame ids found in list file")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    extracted = 0
    with ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for fid in frame_ids:
            member = f"calib/{fid}.txt"
            if member not in names:
                missing.append(fid)
                continue
            data = zf.read(member)
            (out_dir / f"{fid}.txt").write_bytes(data)
            extracted += 1

    out_files = sorted(p.stem for p in out_dir.glob("*.txt"))
    expected = sorted(set(frame_ids))
    actual = sorted(set(out_files))

    print(f"expected_ids={len(expected)}")
    print(f"extracted_files={extracted}")
    print(f"output_txt_count={len(out_files)}")

    if missing:
        print(f"missing_in_zip={len(missing)}")
        print("missing_examples=" + ",".join(missing[:10]))

    extra = sorted(set(actual) - set(expected))
    lack = sorted(set(expected) - set(actual))
    if extra or lack or missing:
        raise RuntimeError(
            "Calibration sync failed strict check: "
            f"extra={len(extra)}, lack={len(lack)}, missing_in_zip={len(missing)}"
        )

    print("strict_check=OK")


if __name__ == "__main__":
    main()
