"""Central configuration + feature/outcome specification for PRO-CABS.

Same structure as PRO-PANN `src/config.py` (SEED, BASE_PATH, DEVICE, feature
groups, outcome registry). What changes is the cohort (cosmetic breast surgery
and/or abdominoplasty, built by `cohort/02_build_cohort.py`), the procedure
feature block, the outcome set and the temporal split.
"""
from pathlib import Path

# --- Reproducibility ---------------------------------------------------------
SEED = 42424242  # same seed as PRO-PANN

# --- Paths -------------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[1]
COHORT_CSV = BASE_PATH / "cosmetic_cohort.csv"

DEVICE = "cpu"

# --- Modelling window + temporal split (strictly by cohort_year) -------------
# The eligible cohort spans 2008-2024, but modelling starts in 2012: secondary
# CPT fields are empty for Plastics before 2010 (combined cases are invisible),
# and unplanned-readmission (READMISSION1-5), REOPERATION1 and PATOS fields
# begin in 2012. The test era is three years (2022-2024) because a single year
# holds only ~600 cases (4-8 events per primary outcome).
#
# Design A (exploratory analysis, results/exploratory/EXPLORATORY_REPORT.md):
# develop on ALL of 2012-2021 -- hyperparameters tuned and probabilities
# calibrated by internal stratified CV -- instead of holding 2020-2021 out as a
# separate calibration era. Learning curves were still rising, and the held-out
# era cost ~1,600 cases and a quarter of the events.
MODEL_START_YEAR = 2012
TRAIN_YEARS = list(range(2012, 2022))   # 2012-2021 (development)
VAL_YEARS = []                          # none: tuning + calibration by internal CV
TEST_YEARS = [2022, 2023, 2024]
CAL_FOLDS = 5                           # internal CV folds for sigmoid calibration

# =============================================================================
#  FEATURE SET  (preoperative-only; same variables as PRO-PANN)
# =============================================================================
NUMERIC_COLS = [
    "age", "height", "weight",
    "pralbum", "prwbc", "prhct", "prplate", "prcreat", "prsodm",
]

ORDINAL_COLS = ["asaclas", "diabetes", "fnstatus2"]
ORDINAL_RANGES = {
    "asaclas": (1, 4),
    "diabetes": (0, 2),
    "fnstatus2": (0, 2),
}

# `sex` is dropped: the cohort is female by definition.
NOMINAL_COLS = ["race_new", "acuity", "anesthes", "surgspec"]

BINARY_COMORBID_COLS = [
    "ethnicity_hispanic", "smoke", "hxcopd", "hxchf", "hypermed", "ascites",
    "dialysis", "discancr", "steroid", "bleeddis", "transfus",
]

# Procedure block (the PRO-PANN co-procedure analogue). Each is a binary CPT
# flag from cohort/02_build_cohort.py; `combined_procedure` = abdominoplasty AND
# any breast procedure (the "mommy makeover" combination studied by Kim et al.).
COPROC_COLS = [
    "abdominoplasty", "breast_augmentation", "mastopexy", "breast_reduction",
    "implant_removal", "capsule_procedure", "other_breast_proc",
    "concurrent_liposuction", "concurrent_umbilical_hernia",
    "other_concurrent_proc", "combined_procedure",
]

BINARY_COLS = BINARY_COMORBID_COLS + COPROC_COLS

FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS + BINARY_COLS

# Dropped by the year-availability audit: discontinued after 2020, so 100%
# missing for 2021-2024 (most of the validation + test era).
YEAR_AUDIT_DROP = ["dyspnea", "wtloss"]

EXCLUDED_COLS = [
    "optime", "wndclas", "tothlos", "dischdest", "yrdeath", "prsepis", "inout",
    "proc_group", "breast_type",
]

# =============================================================================
#  OUTCOMES
# =============================================================================
WOUND_COLS = ["supinfec", "wndinfd", "orgspcssi", "dehis"]
# Kim et al. define surgical/medical complications in their Methods text with
# bleeding under "surgical", but their published Table 3 counts reproduce only
# with surgical = SSI + dehiscence and bleeding under "medical"
# (Research/PRO-CABS/outcome_definition_check/). The Table 3 versions are
# used: they do not overlap, and medical + bleeding performed better in
# development CV (0.799 vs 0.770) with surgical unchanged (0.780 vs 0.781).
SURGICAL_COLS = WOUND_COLS
VTE_COLS = ["othdvt", "pulembol"]
MEDICAL_COLS = [
    "oupneumo", "reintub", "failwean", "renainsf", "oprenafl", "urninfec",
    "cnscva", "cdarrest", "cdmi", "othsysep", "othseshock",
]
MEDICAL_ALL_COLS = MEDICAL_COLS + VTE_COLS + ["othbleed"]   # Kim et al. Table 3 "medical complication"

SYSTEMIC_COLS = VTE_COLS + [c for c in MEDICAL_COLS if c != "urninfec"]

# kind: "single" / "composite" (any of raw cols), "reop", "readmit" (unplanned),
#       "readmit_any" (any or unplanned readmission), "mort",
#       "union" (any of other outcome keys, listed in `cols`)
# Primary outcomes: the best-performing of our candidates and Kim et al.'s four
# (overall, general, surgical, medical), chosen on development CV AUROC
# (outcome_definition_check/OUTCOME_DEFINITION_CHECK.md). Kim et al.'s overall
# (0.708) and general (0.694) lost to any (0.747) and readmission (0.785).
OUTCOME_SPEC = {
    # ---- primary --------------------------------------------------------
    "any":      dict(display="Any Complication",               kind="union",     cols=["surgical", "medical"], deploy=True, primary=True),
    "surgical": dict(display="Surgical Complication (SSI/dehiscence)", kind="composite", cols=SURGICAL_COLS, deploy=True, primary=True),
    "medical":  dict(display="Medical Complication",           kind="composite", cols=MEDICAL_ALL_COLS,  deploy=True,  primary=True),
    "readmit":  dict(display="Unplanned Readmission",          kind="readmit",   cols=[],                deploy=True,  primary=True),
    # ---- secondary (modelled, not deployed) --------------------------------
    "reop":     dict(display="Unplanned Reoperation",          kind="reop",      cols=[],                deploy=False, primary=False),
    "supinfec": dict(display="Superficial SSI",                kind="single",    cols=["supinfec"],      deploy=False, primary=False),
    "vte":      dict(display="Venous Thromboembolism",         kind="composite", cols=VTE_COLS,          deploy=False, primary=False),
    "systemic": dict(display="Systemic Complication (VTE/pneumonia/sepsis/organ)", kind="composite", cols=SYSTEMIC_COLS, deploy=False, primary=False),
    "general":  dict(display="General Complication (Kim et al.: reop/readmission/death)", kind="union", cols=["reop", "readmit_any", "mort"], deploy=False, primary=False),
    "overall":  dict(display="Overall Complication (Kim et al.: surgical/medical/general)", kind="union", cols=["surgical", "medical", "general"], deploy=False, primary=False),
}
# Label-only components (built so unions can reference them; too rare to model:
# 30-day mortality had 0 events 2012-2024).
AUX_OUTCOMES = {
    "mort": dict(display="30-day Mortality", kind="mort", cols=["dopertod", "dischdest"]),
    "readmit_any": dict(display="Any Readmission", kind="readmit_any", cols=[]),
}

ALL_OUTCOMES = list(OUTCOME_SPEC.keys())
PRIMARY_OUTCOMES = [k for k, v in OUTCOME_SPEC.items() if v["primary"]]
DEPLOY_OUTCOMES = [k for k, v in OUTCOME_SPEC.items() if v["deploy"]]

DEPLOY_MODEL_CANDIDATES = ["lr", "lgbm", "xgb"]
ALL_MODELS = ["lr", "lgbm", "xgb", "nn", "stack"]

GATING_MIN_HIGH_BIN_EVENTS = 50

# Risk tiers. PRO-PANN uses four log-spaced bins; on the PRO-CABS test set two of
# the four deployed models were not monotonic across them, and the lowest bin held
# 14-17 patients for two outcomes. PRO-CABS therefore uses three tiers cut at the
# 50th and 90th percentiles of DEVELOPMENT-set predicted risk:
# Low = bottom half, Moderate = 50th-90th, High = top 10%.
BIN_NAMES = ["Low", "Moderate", "High"]
N_BINS = 3
TIER_QUANTILES = (0.50, 0.90)
