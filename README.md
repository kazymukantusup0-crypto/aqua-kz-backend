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

Generate reproducible four-panel visual evaluations and per-patch metrics for the completed model run:

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate_visuals
```

The PNGs, per-patch CSV, and rendering notes are written to `ml/weights/predictions/`.

The best checkpoint is saved to `ml/weights/best_model.pth`. Use
`POST /api/ml/infer/upload` with a raw `(7, 256, 256)` `.npy` tile, or
`POST /api/ml/infer/coordinates` to fetch an authenticated Earth Engine tile and
return a 0/1 drought mask for the frontend.

`app/services/drone_service.py` prepares CSV- and GeoJSON-ready inspection targets only; it does not communicate with a drone.

## Monitoring locations and timelines

Monitoring locations are stored in the local SQLite database at
`data/aqua_kz.db`. Create one with `POST /api/monitoring/locations`, for example:

```json
{"name": "Karaganda demo point", "latitude": 49.8, "longitude": 73.1, "region": "Karaganda"}
```

Run `POST /api/monitoring/locations/{location_id}/analyze` with a year and month
to fetch a calendar-month Earth Engine satellite composite, persist its artifacts,
and save an explainable observation. Set `"enrich": true` to additionally attempt
real daily precipitation and temperature data from the free NASA POWER API; no API
key is required. A month can be unavailable when cloud
filtering leaves no usable imagery; the API reports that condition instead of
silently using another month. The risk score combines available observed stress
indicators and renormalizes weights when inputs are missing. NASA POWER enrichment
reports `partial` quality when rainfall or heat data is available; if it fails or
times out, the request falls back to `segmentation_only`. Retrieve history with
`GET /api/monitoring/locations/{location_id}/timeline` and the current result
with `/latest`. A bounded manual backfill is available at `/backfill`; it
processes at most 24 months and skips existing location-month observations.

The baseline is an approximate empirical mean for the same calendar month over the
most recent ten full years before the requested year, not an official climatological
normal. It is cached in SQLite for 30 days by latitude/longitude rounded to two
decimal places and month. NASA POWER is approximately 0.5 x 0.625 degrees, so do
not request or interpret this as sub-grid precision. Avoid excessive synchronous
request volume, and allow for the roughly 2-3 month lag before near-real-time data
is replaced by higher-quality reprocessed data.

Coordinate analysis with `enrich: true` also derives NDVI and NDMI from the exact
downloaded analysis patch used by segmentation. Sentinel-2 B4/B8/B11 and Landsat
SR_B4/SR_B5/SR_B6 are first mapped to the shared Red/NIR/SWIR1 bands; Sentinel
reflectance uses `0.0001`, while Landsat Collection 2 Level-2 uses `0.0000275`
plus `-0.2`, before the normalized-difference calculations. Satellite vegetation
baselines use the same calendar month over the previous ten years, are cached in
SQLite for 30 days, and are an approximate empirical local baseline rather than an
official climate normal. NDVI anomaly is mapped to vegetation stress from 0 at
non-negative anomaly to 1 at `-0.20`, with linear interpolation and clamping.

To manually verify one enriched observation, open `/docs`, choose the Karaganda
demo location (`49.8, 73.1`), and call
`POST /api/monitoring/locations/{location_id}/analyze` with:

```json
{"year": 2024, "month": 9, "enrich": true}
```

The response should include nullable `ndvi`, `ndmi`, `ndvi_baseline`,
`ndmi_baseline`, `ndvi_anomaly`, `ndmi_anomaly`, and `vegetation_stress` fields.
This request is not automatically re-run for existing months; the controlled
`scripts/resync_karaganda_demo.py` tool is the explicit overwrite path.

Risk is an early-warning support score, not a guaranteed forecast or causal
agricultural diagnosis. The seasonal `/outlook` endpoint currently returns
`forecast_available: false` because no authenticated probabilistic forecast
provider is configured. It explains that monitoring and trend analysis are
available while seasonal outlook requires real probabilistic forecast data.

## National Kazakhstan screening grid

`GET /api/monitoring/national-grid` and `/api/monitoring/national-grid/summary`
serve a separate national screening layer. It is not the detailed monitoring
workflow: a screening cell is a coarse indicator for prioritising inspection,
not a cadastral field boundary, field diagnosis, or agricultural prescription.
Every cell popup and API response carries this limitation. NASA POWER and other
coarse products must not be interpreted at field-scale precision.

The default `kz-grid-v1` grid uses 1.0° × 1.0° cells over edge bounds
40.0–56.0°N and 46.0–88.0°E. Centres are filtered through the bundled
`app/assets/kazakhstan_boundary.geojson`: the Kazakhstan feature from Natural
Earth **1:110m Admin 0 Countries**, which is public domain. The resulting
configuration has 329 centres inside Kazakhstan from 672 bounding-box
candidates. This is deliberately a coarse, configurable screening resolution.

The map returns every configured cell for a period. Only an actual persisted
observation is `ready` and has a risk score. `missing`, `failed`, and `stale`
are data states, never low-risk estimates. For screening cells, the display
logic intentionally distinguishes between a confirmed multi-signal risk and a
preliminary segmentation signal:

- `data_quality == "segmentation_only"` keeps the underlying machine-readable
  `risk_score`, `risk_level`, and `drought_fraction` unchanged, but the UI
  presents it as a preliminary segmentation signal with limited confidence.
- `risk_score`/`risk_level` remain the real persisted values for downstream
  analysis and API consumers; the `assessment_mode` and `confidence_label` fields
  are display-only helper labels and never replace the underlying data.
- `missing` cells stay neutral gray and are never shown as low-risk or ready.

For example:

```text
GET /api/monitoring/national-grid?year=2024&month=9&has_observation=true
GET /api/monitoring/national-grid/summary?year=2024&month=9
```

There is intentionally no public endpoint that starts a country-wide Earth
Engine batch. An operator must use the controlled, sequential CLI instead:

```powershell
# Inspect the full plan: no database writes and no external provider calls.
.\.venv\Scripts\python.exe scripts\prepare_national_grid.py --year 2024 --month 9 --dry-run

# Explicit initial pilot of three cells (only after reviewing the dry run).
.\.venv\Scripts\python.exe scripts\prepare_national_grid.py --year 2024 --month 9 --limit 3 --resume --enrich false

# Resume a selected pilot or interrupted run; ready cells are skipped.
.\.venv\Scripts\python.exe scripts\prepare_national_grid.py --year 2024 --month 9 --resume --cell-id kz-grid-v1-9-23

# Controlled full sequential run, only after an operator has reviewed costs and credentials.
.\.venv\Scripts\python.exe scripts\prepare_national_grid.py --year 2024 --month 9 --resume --enrich false
```

The CLI processes one cell at a time with a 1.5-second pause between requests,
continues after a per-cell failure, records `missing` or `failed` instead of a
fabricated score, and stores a configuration provenance version. `--force` is
an explicit warning-only override for a selected re-analysis; it must not be
used casually. A full 329-cell run has at least about 8 minutes of intentional
rate-limit delay, plus Earth Engine download/inference and optional NASA POWER
latency; cloud filtering, credentials, provider quotas, and unavailable imagery
can increase runtime substantially. It never modifies the existing Karaganda
demo point.
