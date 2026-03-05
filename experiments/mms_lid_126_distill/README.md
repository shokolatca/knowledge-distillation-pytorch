# MMS LID 126 Distillation

## 1. Install modern dependencies

```bash
pip install -r requirements-mms.txt
```

## 2. Prepare manifest

Supported formats: `.jsonl` or `.csv`.
Required fields:
- `id`: unique clip id
- `audio_path`: absolute or relative path to audio file
Optional fields:
- `split`: `train` / `dev`
- `label`: integer class id in `[0, 125]`

If your manifest has no `split` column, set `"train_split": null` and `"dev_split": null` in `params.json`.

Example JSONL row:

```json
{"id":"clip_0001","audio_path":"/abs/path/clip_0001.wav","split":"train","label":17}
```

## 3. Export pseudo labels from ONNX teacher (offline)

```bash
python tools/export_pseudo_labels_onnx.py \
  --manifest data/mms_lid_126/manifest.jsonl \
  --split train \
  --onnx /path/to/mms_lid_126.onnx \
  --output data/mms_lid_126/pseudo_train.npz \
  --sample-rate 16000 \
  --window-sec 2.0 \
  --hop-sec 1.0 \
  --input-format waveform
```

If ONNX expects mel features instead of waveform, set `--input-format logmel`.

## 4. Train student

```bash
python train_mms_lid.py --model_dir experiments/mms_lid_126_distill
```

## 5. Evaluate

```bash
python evaluate_mms_lid.py --model_dir experiments/mms_lid_126_distill --restore_file best
```
