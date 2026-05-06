# Maritime Dehazing / Achelous++ Workspace

This repository contains the Achelous++ codebase and related scripts for water-surface perception with vision-radar fusion. It also includes selected run results (logs and inference outputs) while excluding large datasets.

## Repository Structure
- Achelous-main/: core training, inference, and model code
- frustum_fusion.py: auxiliary script
- val.txt: sample validation list

## Environment
See [Achelous-main/requirements.txt](Achelous-main/requirements.txt).

## Dataset Notes
Large datasets are not included in this repo. Expected dataset folders (not tracked):
- calib/
- radar/
- WaterScenes_Medium/
- WaterScenes-mini/
- WaterScenes-Published/

## Training
Use the training entry point inside Achelous-main:

```bash
cd Achelous-main
python train.py --help
```

## Inference
Run inference from Achelous-main:

```bash
cd Achelous-main
python inference_3d.py --help
```

## Results Included
- Achelous-main/logs*/ (training logs)
- Achelous-main/logs_seg*/ (segmentation logs)
- Achelous-main/export_results/ (exported results)
- Achelous-main/inference_3d_result*.json (sample inference outputs)

## Upstream Project
See the original project documentation in [Achelous-main/README.md](Achelous-main/README.md).
