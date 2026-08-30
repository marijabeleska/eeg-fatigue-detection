# MPD-DF ML service

Python code for inspecting, preprocessing, training, evaluating, and serving the EEG fatigue models.

## Local setup (Windows PowerShell)

```powershell
cd ml-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Inspect the pilot EDF file

```powershell
python scripts\inspect_edf.py
```

The script reads only the EDF metadata (`preload=False`), so it does not load the complete recording into RAM.

## Preprocess the pilot subject

```powershell
python scripts\preprocess_subject.py --subject 01
```

The default pipeline selects `C3 C4 O1 O2`, converts the MPD-DF samples from microvolts to volts, filters to `0.5-40 Hz`, resamples to `128 Hz`, creates non-overlapping 5-second windows that never cross a label transition, and stores the local outputs under `data/processed/participant_01/`.

The explicit microvolt conversion is necessary because the physical-unit text in the MPD-DF EDF header is not decoded correctly by MNE. Keeping all processed EEG in volts follows MNE conventions and prevents incorrect amplitude statistics.

To process several participants sequentially, which keeps RAM usage low, and then
create a compact summary:

```powershell
python scripts\preprocess_all.py --subjects 01 02 03 04 05 06 07 08 09 10
python scripts\summarize_processed.py --subjects 01 02 03 04 05 06 07 08 09 10
```

## Create quality-control plots

```powershell
python scripts\plot_qc.py --subject 01
```

The command creates representative alert/fatigue EEG plots, an averaged PSD comparison, and a relative band-power CSV under `artifacts/qc/participant_01/`. These artifacts are local quality-control outputs and are excluded from Git.

## Download selected MPD-DF participants

```powershell
python scripts\download_mpd_df.py --subjects 01 02 03 04 05 06 07 08 09 10
```

Only raw EEG and annotation files are downloaded. PSG and questionnaire files are not needed for the initial low-channel experiments. Existing files are skipped only after their size and MD5 checksum have been verified.

Processed metadata includes a configurable `quality_flag`. The default marks windows with a maximum channel peak-to-peak amplitude above `200 µV` as `possible_artifact`; it does not delete them.

## Frequency features and baseline benchmark

```powershell
python scripts\extract_features.py --subjects 01 02 03 04 05 06 07 08 09 10
python scripts\benchmark_baseline.py
python scripts\plot_baseline_results.py
```

The feature script calculates relative delta, theta, alpha, and beta power plus
interpretable band-power ratios for every channel. The benchmark uses
leave-one-subject-out evaluation and compares logistic regression with a random
forest, both on all windows and on windows not marked as possible artifacts.

## EEGNet

Run one short pilot fold first:

```powershell
python scripts\train_eegnet.py --test-subjects 01 --epochs 2 --patience 2
```

Run the complete subject-independent benchmark:

```powershell
python scripts\train_eegnet.py --epochs 20 --patience 4
python scripts\plot_model_comparison.py
```

Every fold uses one unseen test subject and a different validation subject for
early stopping. Channel normalization is calculated from training subjects only.
Fold models and metrics are stored under `artifacts/eegnet/`.
