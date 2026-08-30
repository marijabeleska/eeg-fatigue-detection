# Handoff: low-channel EEG fatigue detection

Овој документ може директно да се испрати во нов Codex разговор за проектот да
продолжи од сегашната состојба.

## 1. Цел на проектот

Универзитетски проект за детекција на когнитивен/возачки замор со ниско-канален
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

Dataset-от локално зафаќа приближно:

- `data/raw`: 2,228 MB (20 датотеки: EEG + annotation за 10 лица)
- `data/processed`: 125.7 MB
- `data/features`: 1.7 MB
- `artifacts`: помалку од 1 MB

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

## 6. Neuro-GPT белешка

Менторката предложи да се разгледа Neuro-GPT/foundation models. Официјалниот
Neuro-GPT е pre-trained на TUH EEG со 22 канали, 250 Hz и 2-секундни chunks. Нашиот
pipeline користи 4 канали, 128 Hz и 5 секунди, па pretrained Neuro-GPT не може
директно и научно коректно да се fine-tune-ира без channel/sampling adaptation.

Најразумна идна варијанта е:

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

Работен root на стариот лаптоп:

```text
C:\Users\Admin\Documents\ChatGPT\Faks Project
```

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

## 8. Алатки и верзии на стариот лаптоп

Оперативен систем: Windows; командите се извршувани со PowerShell.

- PyCharm Professional/Community: `2024.2.4`
- Python: `3.13.2`
- PHP CLI преку XAMPP: `8.2.12`, path `C:\xampp\php\php.exe`
- Composer: `2.9.2`
- Node.js: `v22.21.1`
- npm: `10.9.4`
- Git for Windows: `2.48.1.windows.1`
- Laravel skeleton: framework constraint `^12.0`
- Laravel frontend: Vite 7, Tailwind CSS 4, Axios 1.11

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

Транзитивните Python пакети автоматски ќе ги инсталира `pip`; не треба рачно да
се внесуваат сите од `pip freeze`.

Сè уште НЕ се додадени/инсталирани како проектни зависимости:

- FastAPI и Uvicorn (Python prediction API сè уште не е имплементиран);
- Neuro-GPT repository, `transformers` и pretrained Neuro-GPT weights;
- CUDA-enabled PyTorch build (досегашното EEGNet обучување беше на CPU).

## 9. Поставување на нов лаптоп

Инсталирај:

1. Git for Windows.
2. Python 3.13 x64 и додај го во PATH.
3. PyCharm.
4. XAMPP/PHP 8.2 или понов компатибилен PHP 8.x.
5. Composer 2.x.
6. Node.js 22 LTS и npm.

Потоа отвори го целиот `Faks Project` folder во PyCharm и изврши:

```powershell
cd "PATH\TO\Faks Project\ml-service"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Во PyCharm избери interpreter:

```text
Faks Project\ml-service\.venv\Scripts\python.exe
```

Ако новиот лаптоп има NVIDIA GPU, прво провери ја официјалната PyTorch команда
за CUDA-compatible build. Обичната команда од requirements може да работи на
CPU, но не гарантира дека ќе ја користи конкретната GPU/CUDA комбинација.

За Laravel:

```powershell
cd "PATH\TO\Faks Project\web-app"
composer install

# Само ако .env не е пренесен:
Copy-Item .env.example .env
php artisan key:generate

# Ако се користи SQLite и датотеката недостасува:
New-Item database\database.sqlite -ItemType File -Force
php artisan migrate

npm install
npm run build
```

На стариот лаптоп `web-app/vendor` постои, но е нецелосен и недостасува
`vendor/autoload.php`. Затоа на новиот лаптоп мора да се изврши `composer install`.
Laravel функционалноста за EEG сè уште не е имплементирана.

## 10. Пренос на проектот

Најсигурно е целиот root folder да се копира преку надворешен диск, но може да
се прескокнат генерираните dependency папки:

- НЕ мора да се копира `ml-service/.venv` (околу 920 MB со пакетите).
- НЕ мора да се копира `web-app/vendor` бидејќи е нецелосен.
- НЕ мора да се копира `web-app/node_modules`, ако постои.
- По желба може да се прескокне `.idea` и повторно да се отвори проектот.

Треба да се копираат:

- целиот source code;
- `data/raw` ако сакаме 22-channel/Neuro-GPT или повторен preprocessing;
- `data/processed` и `data/features` за веднаш да продолжи ML работата;
- `artifacts` за резултатите, графиците и EEGNet fold моделите;
- `.env` само преку приватен пренос, никогаш во јавен Git repository.

Тековниот Git repository нема commits и сите главни папки се untracked.
Дополнително, root `.gitignore` намерно ги игнорира `data/*`, `artifacts/`,
`ml-service/.venv`, `.idea` и Python cache. Затоа обично Git push нема да ги
пренесе dataset-от, резултатите или моделите; тие мора да се копираат одделно.

Ако raw data не се копира, може повторно да се преземе:

```powershell
cd ml-service
.\.venv\Scripts\Activate.ps1
python scripts\download_mpd_df.py --subjects 01 02 03 04 05 06 07 08 09 10
```

Потоа pipeline-от може целосно да се регенерира:

```powershell
python scripts\preprocess_all.py --subjects 01 02 03 04 05 06 07 08 09 10
python scripts\summarize_processed.py --subjects 01 02 03 04 05 06 07 08 09 10
python scripts\extract_features.py --subjects 01 02 03 04 05 06 07 08 09 10
python scripts\benchmark_baseline.py
python scripts\train_eegnet.py --epochs 15 --patience 3
python scripts\plot_model_comparison.py
```

## 11. Следни задачи

Продолжи по овој редослед:

1. Направи channel benchmark со иста LOSO поставеност:
   - 1 канал: O1 или O2;
   - 2 канали: O1/O2 и споредбено C3/C4;
   - 4 канали: C3/C4/O1/O2 (веќе постои).
2. Одлучи со менторката дали Neuro-GPT мора практично да се имплементира или е
   доволен related-work/feasibility дел.
3. Ако се имплементира Neuro-GPT, направи посебен 22-channel preprocessing и
   прво пробај encoder-only fine-tuning; не менувај го постојниот 4-channel
   dataset/output.
4. По изборот на најсоодветен модел, обучи еден финален deployment модел на сите
   достапни training subjects. LOSO fold моделите не се deployment модел.
5. Имплементирај FastAPI endpoint за prediction и зачувување JSON/CSV резултати.
6. Доврши Laravel dashboard и поврзи го со FastAPI.
7. Прошири го dataset-от на околу 30 испитаници и повтори ја финалната LOSO
   евалуација.

## 12. Почетна порака за нов Codex разговор

Копирај го следново заедно со овој документ:

> Продолжи го проектот опишан во `PROJECT_HANDOFF.md`. Прво провери ја локалната
> структура, Python interpreter-от, достапноста на `data/processed`,
> `data/features` и `artifacts`, без да ги бришеш или регенерираш ако се валидни.
> Потоа продолжи со subject-independent benchmark на бројот и изборот на EEG
> канали. Задржи LOSO без subject leakage, користи balanced accuracy, F1,
> sensitivity, specificity и ROC AUC, и објаснувај ги чекорите на едноставен
> македонски јазик.
