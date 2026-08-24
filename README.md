# AI Drought Monitoring Platform — Backend

This is the local backend for an educational drought-monitoring MVP. It uses only local, **synthetic** CSV data. Its rule-based scoring and machine-learning classifier are demonstrations, not scientifically validated drought forecasts. Do not use this project for operational agricultural, safety, financial, or drone-flight decisions.

## Requirements

- Python 3.11
- No API keys, database, cloud account, satellite provider, or physical drone is required.

## Local installation and run

Run the following commands from the repository root:

```powershell
cd backend
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

On macOS/Linux, create `.venv311` and activate it with `source .venv311/bin/activate` instead.

Open `http://127.0.0.1:8000/docs` for interactive API documentation. The health endpoint is `GET /health`.

## Tests

```powershell
cd backend
pytest -q
```

## Model workflow

`POST /api/train-model` trains a `StandardScaler` + `RandomForestClassifier` pipeline on `app/data/training_data.csv`, then creates these local files:

- `models/drought_model.joblib`
- `models/metrics.json`

Both datasets are synthetic. Generated model files are ignored by Git. Call `POST /api/predict` only after training; otherwise it returns HTTP 409 with guidance.

## Satellite segmentation workflow

The seven-band segmentation pipeline writes 256 x 256 NumPy tiles into
`data/patches/{train,val,test}/{images,labels}`. The channels are Blue, Green,
Red, NIR, SWIR1, SWIR2, and NDVI; each label is a binary VCI drought mask where
VCI is below 35%. The export source is Sentinel-2 L2A plus Landsat 8/9 Level-2
imagery over Akmola and Karaganda during May through August.

Authenticate the server account before requesting an export:

```powershell
earthengine authenticate
earthengine set_project YOUR_EARTH_ENGINE_PROJECT
```

Then use `POST /api/ml/dataset-export` and poll
`GET /api/ml/dataset-export/{job_id}`. Export requires Google Earth Engine
credentials, an Earth Engine-enabled Google Cloud project, and network access.
Alternatively, set `GEE_PROJECT` in `backend/.env`. Train after enough patches have been exported:

```powershell
python -c "from ml.train import train_segmentation; print(train_segmentation())"
```

To run export and training together, use:

```powershell
python -m ml.run_pipeline --year 2024 --max-patches 120 --epochs 20
```

If patches have already been exported, train without re-downloading them:

```powershell
python -m ml.run_pipeline --skip-export --epochs 20
```

The best checkpoint is saved to `ml/weights/best_model.pth`. Use
`POST /api/ml/infer/upload` with a raw `(7, 256, 256)` `.npy` tile, or
`POST /api/ml/infer/coordinates` to fetch an authenticated Earth Engine tile and
return a 0/1 drought mask for the frontend.

`app/services/drone_service.py` prepares CSV- and GeoJSON-ready inspection targets only; it does not communicate with a drone.
