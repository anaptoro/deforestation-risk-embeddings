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

```
docker compose -f docker/docker-compose.yml up -d --build

```

### **2) Authenticate**
```
python -c "import ee; ee.Authenticate(auth_mode='notebook'); ee.Initialize()"
```
### **3) Export training and eval csvs**
```
PYTHONPATH=/app python scripts/export_samples_to_drive.py \
  --bbox=-63.5,-10.5,-61.5,-8.5 \
  --scale 500 \
  --train_years 2018,2019,2020 \
  --n_pos 5000 --n_neg 5000 \
  --unbiased_year 2022 --n_unbiased 30000 \
  --prefix aef_v5 \
  --drive_folder deforestation-risk-exports
```
### **4)Train logistic regression**
```
PYTHONPATH=/app python scripts/train_logit.py \
  --train_csv /app/data/aef_train_balanced_2018_2020_v5.csv \ 
  --unbiased_csv /app/data/aef_unbiased_forest_eval_2022_v5.csv \
  --train_years 2018,2019 \
  --test_year 2020 \
  --out_json models/logit_weights_v5.json
```
### **5)Generate a full Earth Engine Code Editor URL (copy/paste)**
```
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
```

### **6)Use in the Earth Engine Code Editor**

Open the Code Editor.

Paste/open gee/menagerie_loader.js as your script.

Paste the generated URL into the browser address bar (it includes the #... fragment).

The script reads params from the URL fragment and renders:

score/probability layers

Sentinel-2 RGB composites for comparison

### **Follow this if you only want to load the results in gee**
You can simply test it out using this link [here](https://code.earthengine.google.com/f99d62e88cf8822856f896a8d4d0042b#title=Risk66%20logit%20v5;tag=logit66;year=2022;lat=-9.5;lon=-62.5;zoom=9;lo=-10.0;hi=5;roadKm=15;nfMaxKm=5.0;s2m1=7;s2m2=9;s2cloud=60.0;s2Years=2020,2021,2022,2023;b=-2.625581674;w=10.49604201,11.11792296,2.881750663,-1.742704472,-6.205867469,1.07072429,3.870274606,12.29381267,-21.81156267,-21.3376659,18.71999687,-4.493672445,11.49636607,-4.31346941,-28.0044019,-7.552915263,-26.47992487,2.793724924,-8.245010842,-16.16157339,6.81438887,14.95101562,2.836450433,-6.140237865,-17.37485802,-7.823289286,7.634643847,-3.320224022,8.290553763,14.79957287,-3.059269013,0.6107308044,5.753562497,-12.60241098,1.935257108,-13.41428443,-5.041811385,-1.98910115,7.240457196,5.211322518,5.356885362,7.118369858,-8.429068035,-11.47139308,-6.803574381,9.91482752,9.810830721,5.084736678,-6.935314365,10.31480978,6.72819567,-18.33125004,12.50187142,17.05933925,41.30451319,3.503544506,-16.49335902,2.624967836,-17.24139014,-7.814572236,-6.109590986,-22.91546769,-23.73421477,6.206458066,-0.001386822208,-2.461460682e-05):

Or: 

Open the Earth Engine Code Editor
Go to the Earth Engine Code Editor and create a new script.

Paste the loader script
Copy the contents of:

gee/menagerie_loader.js

…into the Code Editor.

Paste the pre-generated URL in your browser

Once the layers are loaded you can switch the parameters and play around to see how it goes

