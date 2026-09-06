# Handoff: low-channel EEG fatigue detection

## 1. Цел на проектот

Проект за детекција на когнитивен/возачки замор со ниско-канален
EEG. Главниот ML pipeline е во Python, а крајниот кориснички интерфејс треба да
биде Laravel веб-апликација. Планираниот тек е:

```text
MPD-DF EEG
  -> Python preprocessing
  -> feature baselines и EEGNet
  -> обучен deployment модел
  -> FastAPI prediction service
  -> Laravel dashboard, графици и fatigue prediction
```

## 2. Dataset и експериментална поставеност

- Dataset: Multimodal Phenotyping Dataset of Driving Fatigue (MPD-DF).
- Официјален Figshare article ID: `28455737`.
- DOI: `10.6084/m9.figshare.28455737.v2`.
- За почетната фаза се користат испитаници `01` до `10`.
- Локално се преземени само raw EEG EDF и annotation TXT датотеките, не PSG и
  questionnaire датотеките.
- Од EEG се користат 4 канали: `C3`, `C4`, `O1`, `O2`.
- Raw sampling rate: `500 Hz`.
- По preprocessing: `128 Hz`, band-pass `0.5-40 Hz`.
- Сигналот се дели на непреклопувачки прозорци од `5 секунди`, односно секој
  прозорец има форма `(4, 640)`.
- Прозорец никогаш не преминува преку annotation/label transition.
- Binary labels: `0 = alert`; оригиналните MPD-DF labels `1-4 = fatigue`.
- EDF unit header е оштетен/нечитлив. Raw вредностите се третираат како
  microvolts и експлицитно се множат со `1e-6` за да се претворат во volts пред
  филтрирање.
- Прозорец со максимална channel peak-to-peak амплитуда над `200 µV` се означува
  како `possible_artifact`, но не се брише.

## 3. Preprocessing резултати

Вкупно се создадени `13,722` прозорци:

- alert: `9,767`
- fatigue: `3,955` (`28.82%`)
- possible artifacts: `651` (`4.74%`)
- нема NaN или infinite вредности

Распределба по испитаник:

| Subject | Windows | Fatigue | Possible artifacts |
|---|---:|---:|---:|
| 01 | 1443 | 8.87% | 1.87% |
| 02 | 1429 | 32.40% | 3.85% |
| 03 | 1440 | 11.32% | 6.32% |
| 04 | 1424 | 68.89% | 1.05% |
| 05 | 650 | 57.85% | 2.77% |
| 06 | 1434 | 20.99% | 5.02% |
| 07 | 1439 | 5.21% | 1.11% |
| 08 | 1439 | 0.28% (само 4 fatigue прозорци) | 0.97% |
| 09 | 1418 | 73.77% | 17.00% |
| 10 | 1606 | 26.03% | 6.35% |

Важни ограничувања: subject 08 има само 4 fatigue прозорци и неговите F1/LOSO
метрики се нестабилни. Subject 09 има значително повеќе possible artifacts.

## 4. Frequency-feature baseline

За секој прозорец се извлечени `36` карактеристики: по 9 за секој од 4-те
канали. Се користат relative delta/theta/alpha/beta power, log total power и
theta/alpha, theta/beta, alpha/beta и `(theta+alpha)/beta` односи.

Евалуацијата е subject-independent leave-one-subject-out (LOSO). Ниту еден
прозорец од test subject не се појавува во training.

| Data | Model | Balanced accuracy | F1 | ROC AUC |
|---|---|---:|---:|---:|
| All windows | Logistic Regression | 0.571 | 0.271 | 0.597 |
| All windows | Random Forest | **0.576** | 0.205 | **0.607** |
| Good only | Logistic Regression | 0.552 | **0.278** | 0.584 |
| Good only | Random Forest | 0.569 | 0.205 | 0.586 |

Отстранувањето на possible-artifact прозорците не го подобри општиот резултат.

## 5. EEGNet

Имплементирана е PyTorch EEGNet-8,2-style мрежа за влез `(batch, 1, 4, 640)`:

- temporal convolution: `F1=8`, kernel length 64
- depthwise spatial convolution преку сите 4 канали
- depth multiplier `D=2`
- separable convolution и `F2=16`
- ELU, batch normalization, average pooling и dropout 0.5
- binary output со BCE-with-logits и class weighting
- вкупно `1,489` trainable параметри

EEGNet исто така е тестиран со 10-fold LOSO. Во секој fold:

- еден цел испитаник е test;
- друг цел испитаник е validation за early stopping;
- останатите лица се training;
- channel mean/std се пресметуваат само од training subjects;
- максимум 15 epochs, patience 3, batch size 64, Adam optimizer.

EEGNet LOSO резултат:

- accuracy: `0.671 ± 0.171` (не е главна метрика поради class imbalance)
- balanced accuracy: `0.558 ± 0.063`
- precision: `0.416 ± 0.279`
- sensitivity/recall: `0.384 ± 0.241`
- specificity: `0.732 ± 0.275`
- F1: `0.302 ± 0.208`
- ROC AUC: `0.583 ± 0.102`

EEGNet има подобар F1 од класичните baselines, но Random Forest има подобра
balanced accuracy и ROC AUC. Во `artifacts/eegnet/all/models/` има 10 fold-specific
модели. Тие се benchmark модели, не се финален deployment модел за API.

## 6. Neuro-GPT -  идно продолжување на проектот

Официјалниот Neuro-GPT е pre-trained на TUH EEG со 22 канали, 250 Hz и 2-секундни chunks. Нашиот
pipeline користи 4 канали, 128 Hz и 5 секунди, па pretrained Neuro-GPT не може
директно и научно коректно да се fine-tune-ира без channel/sampling adaptation.

Идна идеја:

1. EEGNet да остане главниот low-channel модел.
2. Neuro-GPT да биде дополнителен/истражувачки benchmark.
3. Ако се имплементира, прво да се проба pretrained encoder-only стратегијата,
   која била најдобра во оригиналниот труд.
4. За Neuro-GPT веројатно ќе треба повторен preprocessing со 22 канали и 250 Hz,
   или внимателно дефинирана channel-adaptation стратегија.
5. Не треба missing 18 channels само да се пополнат со нули без научно
   образложение.

Референци:

- EEGNet: https://doi.org/10.1088/1741-2552/aace8c
- EEGNet official code: https://github.com/vlawhern/arl-eegmodels
- Neuro-GPT paper: https://arxiv.org/abs/2311.03764
- Neuro-GPT official code: https://github.com/wenhui0206/NeuroGPT
- Neuro-GPT weights: https://huggingface.co/wenhuic/Neuro-GPT

## 7. Проектна структура и скрипти

Главни папки:

```text
Faks Project/
  data/
    raw/participant_01 ... participant_10/
    processed/participant_01 ... participant_10/
    features/frequency_features_4ch_5s.npz
  artifacts/
    processed_subject_summary.csv
    baseline/
    eegnet/
    qc/participant_01/
  ml-service/
    scripts/
    requirements.txt
    README.md
  web-app/
  PROJECT_HANDOFF.md
```

Python скрипти:

- `download_mpd_df.py`: презема EEG + annotations од Figshare; stream download,
  `.part`, size и MD5 проверка.
- `inspect_edf.py`: EDF header/channel/duration inspection без целосно preload.
- `preprocess_subject.py`: цел preprocessing за едно лице.
- `preprocess_all.py`: сериски preprocessing за повеќе лица за ниска RAM употреба.
- `summarize_processed.py`: per-subject class/artifact summary.
- `plot_qc.py`: representative EEG, PSD и band-power QC за еден испитаник.
- `extract_features.py`: 36 frequency features.
- `benchmark_baseline.py`: Logistic Regression/Random Forest LOSO.
- `plot_baseline_results.py`: baseline графика.
- `train_eegnet.py`: EEGNet LOSO training и fold model saving.
- `plot_model_comparison.py`: baseline vs EEGNet графика.

Клучни output датотеки:

- `artifacts/processed_subject_summary.csv`
- `artifacts/baseline/loso_fold_metrics.csv`
- `artifacts/baseline/loso_summary.csv`
- `artifacts/baseline/loso_summary.png`
- `artifacts/eegnet/all/loso_fold_metrics.csv`
- `artifacts/eegnet/all/loso_summary.csv`
- `artifacts/eegnet/all/models/eegnet_test_subject_01.pt` ... `_10.pt`
- `artifacts/eegnet/model_comparison.png`

## 8. Алатки и верзии 

Директните Python зависимости во `ml-service/requirements.txt` се:

```text
mne==1.12.1
matplotlib==3.11.1
numpy==2.5.2
scipy==1.18.1
requests==2.34.2
scikit-learn==1.9.0
torch==2.13.0
```
