# Development, Validation, and Deployment of a Machine Learning Risk Calculator for 30-Day Complications After Cosmetic Breast Surgery and Abdominoplasty

[![Built on ML-Tongue-Pred](https://img.shields.io/badge/built%20on-ML--Tongue--Pred-blue.svg)](https://github.com/AnthonyMatarr/ML-Tongue-Pred)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description
This project implements Logistic Regression, LightGBM, XGBoost, Neural Network, and Stacked Generalization models to predict 30-day postoperative complications in women undergoing cosmetic breast surgery, abdominoplasty, or a combined operation in the American College of Surgeons National Surgical Quality Improvement Program (ACS-NSQIP) dataset.

**Primary outcomes:**
- Any complication (surgical or medical)
- Surgical complication
- Medical complication
- Unplanned readmission

Surgical complication is surgical site infection or wound dehiscence. Medical complication includes bleeding requiring transfusion. Both follow the published counts of the source study (Kim et al., Table 3).

**Secondary outcomes** (modeled and reported in the manuscript, but not deployed):
- Unplanned reoperation
- Superficial surgical site infection
- Venous thromboembolism
- Systemic complication (venous thromboembolism, pneumonia, sepsis, or organ failure)
- General complication (reoperation, readmission, or death)
- Overall complication (surgical, medical, or general complication)

Models are developed with data from 2012-2021, with hyperparameters tuned and probabilities calibrated by internal cross-validation, and validated on a held-out 2022-2024 cohort.

## Building on the Framework of Matar et al.
PRO-CABS builds on the machine learning framework that Matar et al. developed for PRO-TONGUE ([ML-Tongue-Pred](https://github.com/AnthonyMatarr/ML-Tongue-Pred)).

**Carried over from the framework:**
- A preprocessing pipeline fit on training data only. It covers iterative imputation, BMI derived from height and weight, scaling, ordinal clipping, and one-hot encoding.
- Hyperparameter-tuned logistic regression, LightGBM, XGBoost, neural network, and stacked generalization models, each calibrated on a separate validation cohort.
- Lift-based risk stratification into risk tiers, with event rates and lift per tier.
- Patient-level SHAP explanations.
- The Streamlit interface (`base_app.py`, `display_functions.py`, `utils.py`, `shap_utils.py`) and the keep-awake workflow.

**Adapted for cosmetic breast surgery and abdominoplasty:**
- **Cohort:** adult women with a cosmetic postoperative diagnosis (ICD-9 V50.1 / ICD-10 Z41.1) whose procedures were limited to cosmetic breast surgery and/or abdominoplasty, following Kim et al. (ACS-NSQIP 2008-2021) and extended through 2024. Modeling uses 2012-2024, when reoperation, readmission and secondary-procedure fields are recorded consistently, and splits strictly by year.
- **Predictors:** preoperative demographics, comorbidities and laboratory values, plus eleven procedure flags (abdominoplasty; breast augmentation, mastopexy, reduction, implant removal, capsule surgery, other breast procedure; liposuction; umbilical hernia repair; other; and combined abdominoplasty with breast surgery). Dyspnea and weight loss were excluded because ACS-NSQIP has not recorded them since 2021.
- **Outcomes:** 10 outcomes are modeled, and the 4 primary outcomes are deployed. The primary outcomes are the best performers, by development-year cross-validated AUROC, among our candidate outcomes and the source study's four outcomes.
- **Model tuning:** shallow, strongly regularized gradient boosting (tree depth 2-5), and tuning and calibration by internal cross-validation instead of a separate calibration cohort.
- **Deployment artifacts:** aggregate test-cohort summaries replace per-patient predictions (see [Data Privacy](#data-privacy)).

## Associated Risk Calculator
**PRO-CABS** (**P**ostoperative **R**isk of **O**utcomes after **C**osmetic **A**bdominoplasty and/or **B**reast **S**urgery) is a web application that deploys the calibrated model chosen for each of the four primary outcomes (see `app/chosen_models.json`). It places each patient in a **Low, Moderate, or High** risk tier, cut at the 50th and 90th percentiles of predicted risk in the development data, so High means the top 10%. The interface can be found [here](https://pro-cabs.streamlit.app/).

Once all [Installation Steps](#installation) are completed, you can also run the app locally with `.venv/bin/streamlit run app/base_app.py`.

### Features
- Enter the patient in four sections: Patient, Health status, Preoperative labs, and Operation (abdominoplasty, breast procedures, and procedures performed with them). BMI is calculated from height and weight.
- Laboratory values are optional; values marked not available are imputed by the fitted preprocessing pipeline.
- Choose which of the 4 primary outcomes to estimate (all are selected by default), then select **Estimate risk**.
- Results show a summary card per outcome and a detailed tab for each:
  - **Calibrated risk** probability
  - **Risk tier**: Low, Moderate, or High, shown on a tier scale with cut points
  - **Observed event rate** among 2022-2024 test-cohort patients in each risk tier
  - **Percentile ranking** of model output relative to the 2022-2024 cohort
  - **Contributing factors** via patient-level SHAP explanation values
- The sidebar summarizes the models, defines each outcome, and states who the models apply to.

## Project Layout
### Included directories
- `app/`: all code and artifacts for the interface
  - `base_app.py`: source code for the calculator interface
  - `display_functions.py`: helper file for the modules displayed on the interface
  - `utils.py`: helper functions for risk bins, observed rates, and percentiles
  - `shap_utils.py`: helper functions for SHAP analysis
  - `models/calibrated/`: calibrated model for each primary outcome
  - `preprocessors/pipeline.joblib`: fitted preprocessing pipeline
  - `shap_explainers/`: per-outcome SHAP explainers and the reduced feature list
  - `bin_thresholds/`: risk-bin cut points
  - `population_summary/`: de-identified 2022-2024 test-cohort summaries
- `src/`: the minimal modules the app and the pickled pipeline need (`config`, `preprocess`, `shap_utils`)
- `tools/`: script to rebuild the deployment artifacts and de-identified summaries from the private project
- `.streamlit/`: interface styling. It stays at the repository root for Streamlit Community Cloud.
- `.github/workflows/keep_awake.yml` and `keep_awake.py`: a scheduled job that keeps the hosted app awake
- `licenses/`: the original MIT notice for ML-Tongue-Pred

### Not included in this repository
- `data/`: raw and processed ACS-NSQIP data
- `models/`: raw trained and calibrated models for every candidate model type
- `results/`: figures and tables
- The analysis pipeline: cohort extraction, cleaning, preprocessing, tuning, training, risk stratification, evaluation, and drift analysis

## Installation

### Prerequisites
Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your system.

### Steps (run all in command prompt/terminal)
1. Clone this repository:
```
git clone https://github.com/jj-leee/pro-cabs.git
cd pro-cabs
```
2. Create a Python 3.12 environment from the pinned requirements:
```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```
3. Run the headless check, which prints `DEPLOY SMOKE TEST: PASS`:
```
.venv/bin/python smoke_test.py
```
4. Launch the calculator:
```
.venv/bin/streamlit run app/base_app.py
```

## Data Privacy
ACS-NSQIP Participant Use Files may not be redistributed. This repository contains no patient-level data:
- Per-patient test predictions are replaced by aggregate summaries: the observed event rate in each risk bin, a percentile cut-point lookup, and the test-cohort size.
- Tree-model SHAP explainers use the path-dependent algorithm and store no background data.
- Any linear-model SHAP explainer stores only the training mean and covariance.
- `.gitignore` blocks data files (`*.parquet`, `*.csv`, `*.xlsx`, ...), and `smoke_test.py` fails if any are present.

## Data Source
American College of Surgeons National Surgical Quality Improvement Program and the hospitals participating in the ACS NSQIP are the source of the data used herein; they have not verified and are not responsible for the statistical validity of the data analysis or the conclusions derived by the authors.

## License
MIT. See `LICENSE`; the original ML-Tongue-Pred notice is in `licenses/`.

*For research and clinical decision support only. Estimates do not replace clinical judgment.*
