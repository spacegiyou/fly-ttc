# Manual Nexar train placement

Only the official [Hugging Face Nexar dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)
or [Kaggle Nexar Collision Prediction train split](https://www.kaggle.com/competitions/nexar-collision-prediction/data)
is supported. Do not put synthetic or substitute dataset videos here. Dataset rights remain governed by
the upstream Nexar Open Data License; do not redistribute videos from this repository.

The download command saves the upstream `LICENSE` and `README.md`, two train metadata CSVs,
a pinned commit and exact source paths, and local SHA256 / size / mtime in manifests.
The default is 100 videos (50 positive + 50 negative), seed 0. Existing subset IDs and the
source commit remain frozen on repeat downloads. `HF_TOKEN`, if needed, is read only from
the environment; never put tokens in CSV, config or commands saved in reports.

```bash
python scripts/download_nexar.py --metadata-only
python scripts/download_nexar.py --subset 100 --seed 0
```

For offline use, preserve the official HF directory layout under `data/raw/nexar/`:

```text
train/positive/metadata.csv
train/positive/<exact file_name from metadata.csv>.mp4
train/negative/metadata.csv
train/negative/<exact file_name from metadata.csv>.mp4
```

The observed upstream metadata columns are `file_name,time_of_event,time_of_alert,
light_conditions,weather,scene,time_to_accident`. Positive/negative labels derive from
the official directory labels (1/0); no extra accident annotations are invented.

```bash
python scripts/build_manifest.py --source data/raw/nexar --subset 100 --seed 0
```

For Kaggle, put the official `train.csv` under `data/raw/nexar/` and extracted MP4s under
`data/raw/nexar/train/`. Required CSV columns are `id,target,time_of_alert,time_of_event`.
The builder first tries the documented `train/{id:05d}.mp4` path and then searches actual
MP4 stems. Ambiguous mappings fail. A completely unmatched video directory fails, and
individual missing files are retained so evaluation can record and skip only those IDs.

```bash
python scripts/build_manifest.py --source data/raw/nexar/train.csv --subset 100 --seed 0
```

For nonstandard filenames, supply an explicit normalized CSV with
`video_id,label,video_path,time_of_alert,time_of_event,scene,weather,light_conditions`.
Paths in a source or generated manifest are relative to that CSV's parent directory.
Missing context categories can be `Unknown`; missing timestamps must remain empty.
Both timestamps must be empty for label 0; inconsistent negative annotations are rejected.
The runner records missing files rather than creating data or guessing annotations.

A failed network download prints the exact retry command and stops. Any completed
files and the fixed subset manifest remain available. No test split is used for lead time.
Synthetic fixtures belong only in `tests/` and `outputs/synthetic/`.
