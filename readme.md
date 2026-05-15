# CWL-026-001: Coastal Wetland Carbon Risk Framework
**CSC 580 — Advanced Data Mining and Fusion**
**Louisiana Tech University | Spring 2026**

---
# CAUTION
There is a lot of data in this RFP, a lot of it is too large to be uploaded
to GitHub. All other data needed for the pipeline that is not easily accessible
will be included in a separate .zip file that should be unzipped.
The contents should be placed in the /data folder.


## Project Overview

A spatially continuous, updateable data mining framework that fuses
ground-based CRMS monitoring data with Sentinel-2 satellite imagery
to produce wall-to-wall carbon stock estimates and parcel-level wetland
loss predictions across Louisiana's Mississippi River Deltaic Plain.

Addresses RFP requirements:
1. Wall-to-wall carbon stock estimation (filling gaps between 266 stations)
2. Parcel-level wetland-to-open-water conversion prediction
3. Multi-source data integration (CRMS + Sentinel-2 + Landsat + hydro)
4. Human-interpretable outputs for coastal restoration planners

---

## Full Execution Order

```
01_gee_sentinel2_pipeline.py        → Sentinel-2 spectral features (GEE cloud)
02_crms_preprocessing.py            → Clean + merge all CRMS ground data
fix_loss_target.py                  -> correctly merges a recent_land_loss
fix_integrate.py                    -> adds to the fused final database
03_gaussian_process_regression.py   → Train carbon stock model
04_fpgrowth_pattern_mining.py       → Mine co-occurrence rules
05_c45_classification_temporal_roc.py → Train loss prediction model
07_spatial_prediction.py            → Wall-to-wall coast predictions
sample_wall.py
display_wall.py
get_high_risk.py
06_dashboard.py                     → Launch interactive dashboard
```

### Routine Update Order (new data arrives)
```
02_crms_preprocessing.py                        → merge new CRMS data
fix_loss_target.py                              → recompute recent_land_loss
fix_integrate.py                                → merge into fused dataset
03_gaussian_process_regression.py --rescore     → re-score, no retrain
04_fpgrowth_pattern_mining.py                   → re-mine patterns (always full re-run)
05_c45_classification_temporal_roc.py --rescore → re-score, no retrain
07_spatial_prediction.py                        → update wall-to-wall maps
sample_wall.py && display_wall.py && get_high_risk.py → rebuild display files
streamlit run 06_dashboard.py                   → dashboard reflects new results
```

### Annual Retraining (or after major storm event)
```
Run all scripts in full execution order above (no --rescore flags)
```

### Location Structure for Coordinates
```
crms_all_stations_coords.csv (425 stations)
        ↓
GEE extracts Sentinel-2 at ALL 425 locations
        ↓
Split into two groups:
  ├── CRMS stations (395) → full pipeline with hydro → training data
  └── BA/TE/etc (30)     → spectral only
```

---

## Project File Structure

```
CWLRFP/
├── README.md
├── requirements.txt
│
├── ── PIPELINE SCRIPTS ──
├── 01_gee_sentinel2_pipeline.py           Sentinel-2 GEE export (run in GEE environment)
├── 02_crms_preprocessing.py               CRMS data cleaning + multi-source fusion
├── fix_loss_target.py                     Compute recent_land_loss from loss timeseries
├── fix_integrate.py                       Merge recent_land_loss into fused dataset + retrain C4.5
├── 03_gaussian_process_regression.py      Carbon stock GPR model (habitat-stratified)
├── 04_fpgrowth_pattern_mining.py          FP-Growth + Apriori pattern mining
├── 05_c45_classification_temporal_roc.py  C4.5 loss classifier + temporal ROC
├── 07_spatial_prediction.py               Wall-to-wall coast predictions
├── sample_wall.py                         Stratified sample of pixel predictions
├── display_wall.py                        Dashboard-sized display subset
├── get_high_risk.py                       High-risk unmonitored parcels
├── download_ccap.py                       C-CAP marsh classification download + processing
├── integrate_ornl_baustian.py             Baustian + ORNL DAAC carbon label integration
│── 06_dashboard.py                        Main Streamlit dashboard (4 tabs)
│
│
├── hurricaneImplementation/
│   ├── hurricane_tab.py                   Hurricane impact tab (not currently connected)
│   └── storm_events.py                    Storm registry + scenario engine
│
├── ── DATA (inputs — place in data/) ──
├── data/
│   ├── crms_all_stations_coords.csv       Station coordinates (425 stations)
│   ├── crms_marsh_class.csv               Marsh habitat type by station-year
│   ├── crms_accretion_rates.csv           Vertical accretion rates
│   ├── crms_bulk_density.csv              Soil bulk density
│   ├── crms_percent_organic.csv           Soil percent organic matter
│   ├── crms_land_water.csv                1km land/water classification (loss labels)
│   ├── crms_hydro_averages.csv            Hydro averages (annual, all years)
│   ├── crms_percent_flooded.csv           Percent time flooded per station-year
│   ├── crms_belowground_biomass.csv       Belowground live/dead biomass by depth
│   ├── baustian_longterm_carbon.csv       Baustian et al. (2021) carbon labels
│   ├── deltax_soil_properties.csv         ORNL DAAC Delta-X soil carbon
│   ├── crms_sentinel2_features.csv        GEE station-level spectral export
│   ├── deltaic_plain_spectral_v2_2023_07.tif  Wall-to-wall spectral + aux raster (preferred)
│   └── deltaic_plain_spectral_2023_07.tif     Wall-to-wall spectral raster (fallback)
│
│   ── PROCESSED (generated by pipeline) ──
│   ├── crms_master.csv                    Merged station-level dataset
│   ├── fused_dataset_final.csv            Station-month fused dataset (primary model input)
│   ├── crms_recent_loss.csv               Recent land loss per station (from fix_loss_target.py)
│   ├── crms_freshwater_intermediate.csv   Habitat split
│   ├── crms_brackish.csv                  Habitat split
│   ├── crms_saline.csv                    Habitat split
│   ├── crms_marsh_timeseries.csv          Time-varying marsh type
│   ├── crms_loss_timeseries.csv           Year-by-year land/water labels
│   ├── crms_hydro_station_year.csv        Hydro aggregated to station-year
│   ├── crms_flooded_station_year.csv      Percent flooded aggregated to station-year
│   ├── carbon_training_labels.csv         Enriched carbon labels (from integrate_ornl_baustian.py)
│   └── carbon_validation_set.csv          Independent validation set
│
├── ── MODEL OUTPUTS ──
├── models/
│   ├── gpr_model_{habitat}.pkl            Trained GPR per habitat
│   ├── gpr_scaler_{habitat}.pkl           Feature scaler per habitat
│   ├── gpr_features_{habitat}.pkl         Selected feature list per habitat
│   ├── c45_model.pkl                      Trained C4.5 classifier
│   ├── c45_scaler.pkl
│   └── c45_features.pkl
│
├── ── RESULTS ──
├── results/
│   ├── gpr_results.csv                    GPR R² and RMSE per habitat
│   ├── shap_{habitat}.csv                 Feature importance rankings
│   ├── apriori_itemsets.csv               Apriori baseline itemsets
│   ├── fpgrowth_itemsets.csv              FP-Growth frequent itemsets
│   ├── association_rules.csv              All association rules mined
│   ├── land_loss_rules.csv                Rules predicting recent_loss_YES
│   ├── c45_rules.txt                      Plain-language if-then rules
│   ├── c45_cv_results.csv                 Spatially blocked CV results
│   ├── temporal_roc_results.csv           ROC-AUC per time horizon
│   ├── rfp_classifier_report.csv          Precision/recall/F1 report
│   ├── wall_to_wall_predictions.csv       Full pixel predictions (07)
│   ├── wall_to_wall_sample.csv            Stratified sample (sample_wall.py)
│   ├── wall_to_wall_display.csv           Dashboard display subset (display_wall.py)
│   ├── high_risk_unmonitored.csv          High-risk pixels far from stations (get_high_risk.py)
│   └── high_risk_parcels.geojson          High-risk GeoJSON export (07)
│
└── ── FIGURES ──
    figures/
    ├── shap_*.png                         SHAP feature importance plots
    ├── gpr_predictions_*.png              Predicted vs observed plots
    ├── temporal_roc.png                   Temporal ROC curves
    ├── c45_tree.png                       Decision tree visualization
    └── rules_*.png                        Association rule scatter plots
```

---

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Authenticate Google Earth Engine
```bash
earthengine authenticate
python -c "import ee; ee.Initialize(project='cwlrfp'); print('GEE connected')"
```

---

## Data Sources

| Dataset | Source | Status | Used For |
|---------|--------|--------|----------|
| CRMS Station Coordinates | lacoast.gov/crms | Downloaded | Station locations |
| CRMS Marsh Class | lacoast.gov/crms | Downloaded | Habitat type (time-varying) |
| CRMS Accretion Rates | lacoast.gov/crms | Downloaded | GPR target variable |
| CRMS Bulk Density | lacoast.gov/crms | Downloaded | Carbon stock estimation |
| CRMS Percent Organic | lacoast.gov/crms | Downloaded | Carbon stock estimation |
| CRMS Land/Water 1km | lacoast.gov/crms | Downloaded | Loss labels (C4.5 target) |
| CRMS Hydro Averages | lacoast.gov/crms | Downloaded (2005-2026 yearly) | Salinity, flood depth, tidal amplitude |
| Sentinel-2 Level-2A | ESA via GEE | Exported | Spectral features (NDVI/NDWI/EVI) |
| Sentinel-2 Spatial Grid | ESA via GEE | Exported | Wall-to-wall prediction surface |
| USGS Landsat | Via CRMS land/water | Indirect | Binary loss labels |
| NASA ORNL DAAC | daac.ornl.gov | Downloaded |  carbon cycle products | unable to rectify coordinates returned with sample coordinates |

---

## Methods

| Task | Method | Reference | Status |
|------|--------|-----------|--------|
| Carbon stock estimation | Gaussian Process Regression (habitat-stratified) | Chenevert & Edmonds (2024) | Ready to run |
| Baseline pattern mining | Apriori + prefix tree/hashing | Class methods | Ready to run |
| Primary pattern mining | FP-Growth | Class methods | Ready to run |
| Coast-wide pattern mining | Map-Reduce FP-Growth | Class methods | Ready to run |
| Temporal precursor detection | Sequential temporal mining | Class methods | Ready to run |
| Spatial propagation detection | Spatio-temporal mining | Class methods | Ready to run |
| Loss classification | C4.5 Decision Tree | Class methods | Ready to run |
| Validation | Temporal ROC + Spatially Blocked CV | Class methods | Ready to run |
| Explainability | SHAP values | Post-training | Ready to run |
| Wall-to-wall prediction | GPR + C4.5 applied to GEE raster | This project | Ready to run |

---

## Key Design Decisions

### Preprocessing follows papers exactly
- Outlier removal: 75th percentile + 1.5x IQR (Chenevert & Edmonds 2024)
- Swamp communities excluded (no sedimentologic data)
- Backward elimination feature selection p < 0.05
- 5-fold spatially blocked cross-validation × 100 runs
- Marsh type treated as time-varying (not static) — 54% of sites changed type since 1949

### GEOID datum change handled
- CRMS hydro data uses GEOID99 pre-2013, GEOID12B post-2013
- Water elevation uses NAVD88-corrected values where available
- Salinity and flood depth (relative measures) unaffected

### Storm events separated from chronic conditions
- 11 named storms flagged with impact zones (1992-2021)
- storm_year binary feature added to all models
- FP-Growth mines storm and chronic rules separately

### Continuous update architecture
- Data ingestion separated from model retraining
- Sentinel-2: automated via GEE (new imagery always available)
- CRMS hydro: quarterly manual download → paste new rows into crms_hydro_averages.csv → re-run 02_crms_preprocessing.py
- Model re-scoring: --rescore flag (fast, no retraining)
- Model retraining: annually or after major storm events

### Wall-to-wall predictions
- Models trained on 323 CRMS stations
- Applied to every 100m pixel in Sentinel-2 spatial grid
- Produces continuous carbon stock and loss probability maps
- High-risk parcels exported as GeoJSON for dashboard map layer

---

## Dashboard

### Launch
```bash
streamlit run 06_dashboard.py
```

### Tabs
| Tab | Description | Audience |
|-----|-------------|----------|
| Risk Analysis | Interactive map (station or wall-to-wall) + station detail panel + FP-Growth rules | Planners |
| Investment Priorities | Stations ranked by carbon loss avoided × loss probability + charts | Program officers |
| Field Verification Queue | Stations flagged for priority field visit (loss_prob > 0.5 AND NDVI < 0.4) | Field coordinators |
| Regional Overview | Coast-wide distribution charts and basin-level summary table | All stakeholders |

Note: A Hurricane Impact tab module exists in `hurricaneImplementation/hurricane_tab.py`
but is not currently wired into the dashboard. It can be added as a fifth tab.

### Station Search
Type any station ID (e.g. CRMS0033) in the sidebar search box.
Map zooms to station and detail panel opens automatically.

---

## Required Papers

1. **Thomas et al. (2019)** — Sentinel-2 GEOBIA classification, Louisiana wetland biomass mapping, 90.5% accuracy. Provides: preprocessing workflow, segmentation approach, NDVI/NDWI/EVI validation.

2. **Baustian et al. (2021)** — Soil carbon accumulation rates at 24 CRMS sites. Provides: carbon stock training labels, habitat-specific carbon values, marsh type temporal instability finding.

3. **Chenevert & Edmonds (2024)** — Gaussian Process models on 266 CRMS stations. Provides: preprocessing pipeline, tidal amplitude + flood depth as key predictors, GPR as validated modeling approach, SHAP explainability.

---

## Proposal Sections Status

| Section | Description | Status |
|---------|-------------|--------|
| A1 | Cover Page | Not started |
| A2 | Problem Statement + FROM→TO | Drafted |
| A3 | Proposed Approach & Innovation | Drafted (revised twice) |
| A4 | Data Access & Feature Engineering | Outlined |
| A5 | Hypothesis Framework & Validation | Outlined |
| A6 | Risk Assessment | Not started |
| A7 | Scope & Timeline | Not started |
| A8 | Deliverables & Outcomes | Not started |
| A9 | References | Not started |
| A10 | Related Work Survey (grad) | Not started |
| A11 | Formal Experimental Design (grad) | Not started |
| A12 | Reproducibility Plan (grad) | Not started |

---

## Known Limitations / Future Work

- Monthly hydro averages unavailable via CRMS portal (server limitation) — using yearly averages as proxy
- NASA ORNL DAAC carbon cycle products not yet integrated
- Wall-to-wall marsh type classification uses NDVI thresholds as proxy — Landsat vegetation classification planned for Phase 2
- Storm impact modeling uses distance/surge heuristic — ADCIRC hydrodynamic integration planned for Phase 2
- Chenier Plain extension planned for Phase 2 (current scope: Deltaic Plain only)
