# deforestation-risk-embeddings

Train a simple linear deforestation-risk model on top of Google Earth Engine **Annual Satellite Embeddings (AEF)** plus two “frontier context” features:

- `dist_to_nonforest_m` (from MODIS land cover forest/non-forest transitions)
- `dist_to_road_m` (from GRIP4 roads)

** those frontier context features are hard coded, and they will work only for South america, it can be chanegd though. Also, they dont increase acc a lot, so, for future experiments they may can be removed.

Outputs:
- CSV training/eval tables exported from Earth Engine to **Google Drive**
- Logistic regression weights exported as JSON
- A copy/paste **Earth Engine Code Editor URL** (fragment params) to visualize scores + Sentinel-2 RGB

---

## Usage Guide
### 0) You must have a gee account

### Follow this if you want to train your own regression

### 1) Build and start
From the repo root:

```bash
docker compose -f docker/docker-compose.yml up --build

### 2) Authenticate
python -c "import ee; ee.Authenticate(auth_mode='notebook'); ee.Initialize()"

### 3) Export training and eval csvs

PYTHONPATH=/app python scripts/export_samples_to_drive.py \
  --bbox=-63.5,-10.5,-61.5,-8.5 \
  --scale 500 \
  --train_years 2018,2019,2020 \
  --n_pos 5000 --n_neg 5000 \
  --unbiased_year 2022 --n_unbiased 30000 \
  --prefix aef_v5 \
  --drive_folder deforestation-risk-exports

### 4)Train logistic regression

PYTHONPATH=/app python scripts/train_logit.py \
  --train_csv /app/data/aef_train_balanced_2018_2020_v5.csv \ 
  --unbiased_csv /app/data/aef_unbiased_forest_eval_2022_v5.csv \
  --train_years 2018,2019 \
  --test_year 2020 \
  --out_json models/logit_weights_v5.json

### 5)Generate a full Earth Engine Code Editor URL (copy/paste)

PYTHONPATH=/app python -m src.modeling.export_weights \
  --weights models/logit_weights_v5.json \
  --title "Risk66 logit v5" \
  --tag "logit66" \
  --year 2022 \
  --lat -9.5 --lon -62.5 --zoom 9 \
  --lo -10 --hi 10 \
  --roadKm 100 --nfMaxKm 30 \
  --s2m1 7 --s2m2 9 --s2cloud 60 \
  --s2Years "2020,2021,2022,2023" \
  --print url


### 6)Use in the Earth Engine Code Editor

Open the Code Editor.

Paste/open gee/menagerie_loader.js as your script.

Paste the generated URL into the browser address bar (it includes the #... fragment).

The script reads params from the URL fragment and renders:

score/probability layers

Sentinel-2 RGB composites for comparison