# PRC Data Challenge 2026: Taxi-Out Time Prediction

Predicting taxi-out time (off-block to take-off) for departures from 10 major European airports, for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/) run by EUROCONTROL's Performance Review Commission and the OpenSky Network.

**Result:** XGBoost on leakage-safe off-block features, **308.6 s RMSE** on four held-out months (Jan, Feb, Jun, Jul 2025) against 403.7 s for an airport + runway median baseline.

## Approach in one paragraph

Every feature uses only information known at off-block, anchored on the planned off-block time `AOBT_3_flt` (scheduled time when missing). Features cover recent traffic (departures and arrivals in the last 15/30/60 min), departure delay, METAR weather, a leave-one-month-out typical and unimpeded taxi time per airport + stand + runway, and stand-to-runway distance from OpenStreetMap. The actual take-off time is in the ranking file but is **deliberately excluded**: take-off minus `AOBT_3_flt` reconstructs the target (see `FINAL_PHASE_HANDOVER.md` §5).

## Repository contents

| File | Purpose |
|---|---|
| `enrichment.py` | Builds the feature files from the challenge data + external data |
| `PRC_Eurocontrol_ML_Pipeline_adapted.ipynb` | Model training, evaluation, tuning and prediction (clean version) |
| `PRC_Eurocontrol_ML_Pipeline_adapted_colab_run.ipynb` | Same notebook with outputs from the 7 Oct 2026 run (`PRC_Eurocontrol_ML_Pipeline.ipynb` is an identical copy) |
| `Kamran_Cleaning_Notebook_Documented.ipynb`, `clean_taxi_data.py` | Data cleaning and quality flags (exploratory; not needed to reproduce the submission) |
| `Data_analysis_and_EDA .ipynb` | Exploratory data analysis |
| `FEATURE_DICTIONARY.md` | Every feature, its meaning and source |
| `HANDOVER.md`, `FINAL_PHASE_HANDOVER.md` | Design decisions, checks, ablations and generalisation results |
| `weather_cache.parquet`, `runways_10_airports.csv`, `osm_stands.csv` | External inputs (cached) |
| `fetch_osm_stands.py` | Re-downloads `osm_stands.csv` from OpenStreetMap |
| `enriched_features_*.parquet` | Feature files produced by `enrichment.py` |
| `submission_final_phase.parquet`, `ranking_predictions.parquet` | Final predictions |

## How to reproduce

1. Install dependencies: `pip install -r requirements.txt` (Python 3.11+).
2. Download the challenge files (`training_2025-*.parquet`, `final_ranking.parquet`, `final_submitting.parquet`) from the challenge bucket into this folder. Rename the last two to `ranking.parquet` and `submitting.parquet`. These files are not redistributed here.
3. Run `python enrichment.py` to produce `enriched_features_train.parquet` and `enriched_features_ranking.parquet`.
4. Run `PRC_Eurocontrol_ML_Pipeline_adapted.ipynb` top to bottom. It trains the model, reports holdout results, refits on all of 2025 and writes `submission_final_phase.parquet`. Set the `PRC_DATA_DIR` environment variable if the data sits in another folder.

Training runs on CPU; set `device='cuda'` in the notebook to use a GPU.

## Results

| Held-out months (2025) | Model RMSE (s) | Baseline RMSE (s) |
|---|---|---|
| Jan + Jul | 312.9 | 408.5 |
| Feb + Jun | 304.0 | 398.6 |
| All four | 308.6 | 403.7 |

Most important feature: `typical_taxi_sec` (44% of split gain). Largest error source: snow (707 s RMSE on snowy departures vs 306 s otherwise).

## Data sources

- PRC Data Challenge 2026 data: https://prc-data-challenge-2026.netlify.app/data.html
- Iowa Environmental Mesonet (IEM) ASOS/METAR: https://mesonet.agron.iastate.edu/request/download.phtml
- OurAirports: https://ourairports.com/data/
- OpenStreetMap (© OpenStreetMap contributors, ODbL) via Overpass API: https://www.openstreetmap.org/copyright

## Team
trustworthy-zipper

## Members 
- Kabir Ansari - BSc Business Information Systems, University of Westminster

- Maham Baber - BSc Data Science & Analytics, University of Westminster

- Oscar Denche - BSc Data Science & Analytics, University of Westminster

- Anahita Hessami - MSc Data Science & Analytics, University of Westminster

- Harsh Narendra - BSc Computer Science, University of Westminster

- Kamran Rzayev - MSc Data Science & Analytics, University of Westminster

- Elisa Schezzini - BSc Data Science & Analytics, University of Westminster

- Mihai-Catalin Vasile - MSc Data Science & Analytics, University of Westminster 

## License

GNU General Public License v3.0, see `LICENSE`.
