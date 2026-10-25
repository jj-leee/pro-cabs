"""Input collection + per-outcome result display (adapted from ML-Tongue-Pred app)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config as appcfg
import utils as util
import shap_utils as appshap
from src import config as srccfg
from src.preprocess import remove_prefix

ABDO_LABELS = {
    "abdominoplasty": "Abdominoplasty",
}
BREAST_LABELS = {
    "breast_augmentation": "Augmentation (implant)",
    "mastopexy": "Mastopexy",
    "breast_reduction": "Reduction",
    "implant_removal": "Implant removal",
    "capsule_procedure": "Capsulotomy / capsulectomy",
    "other_breast_proc": "Other breast procedure",
}
ADJUNCT_LABELS = {
    "concurrent_liposuction": "Liposuction",
    "concurrent_umbilical_hernia": "Umbilical hernia repair",
    "other_concurrent_proc": "Other (unlisted skin procedure)",
}
COMORBID_LABELS = {
    "smoke": "Current smoker (within 1 yr)", "hypermed": "Hypertension on medication",
    "hxcopd": "COPD", "hxchf": "Congestive heart failure",
    "dialysis": "Dialysis", "ascites": "Ascites",
    "steroid": "Chronic steroid use", "bleeddis": "Bleeding disorder",
    "transfus": "Preoperative transfusion", "discancr": "Disseminated cancer",
}

OUTCOME_LABELS = {
    "any": "Any complication",
    "surgical": "Surgical complication",
    "medical": "Medical complication",
    "readmit": "Unplanned readmission",
}
OUTCOME_DEFS = {
    "any": "A surgical complication or a medical complication (see their definitions) "
           "within 30 days.",
    "surgical": "Superficial incisional, deep incisional, or organ/space surgical site "
                "infection, or wound dehiscence, within 30 days.",
    "medical": "Bleeding requiring transfusion, deep vein thrombosis or pulmonary "
               "embolism, pneumonia, unplanned reintubation, ventilation over 48 hours, "
               "renal insufficiency or failure, urinary tract infection, stroke, cardiac "
               "arrest, myocardial infarction, or sepsis or septic shock, within 30 days.",
    "readmit": "Unplanned hospital readmission within 30 days.",
}
MODEL_NAMES = {"lgbm": "LightGBM", "xgb": "XGBoost", "lr": "logistic regression"}

TIER_COLORS = ["#2f7d4f", "#c4621a", "#c21615"]
TIER_BADGES = ["green", "orange", "red"]

CSS = """
<style>
.block-container {max-width: 1200px; padding-top: 2.5rem;}
.sec-title {font-size: 1.05rem; font-weight: 600; margin: 0 0 0.35rem 0;}
.sub-label {font-size: 0.875rem; font-weight: 600; color: #4b5563; margin: 0.6rem 0 0.1rem 0;}
.card-label {font-size: 0.875rem; color: #4b5563; line-height: 1.3; min-height: 2.6em;}
.calc-label {font-size: 0.875rem; line-height: 1.6; margin-bottom: 0.25rem;}
.calc-value {font-size: 1rem; font-weight: 600; padding: 0.4rem 0 0.1rem 0;}
.calc-note {font-size: 0.75rem; color: #6b7280;}
div.risk-value {font-size: 2rem; font-weight: 600; line-height: 1.15; margin: 0.15rem 0 0.35rem 0;}
.tier-scale {display: flex; gap: 4px; margin: 0.4rem 0 0.9rem 0;}
.tier {flex: 1; border-radius: 6px; padding: 7px 9px; background: #f3f4f6;
       color: #4b5563; font-size: 0.78rem; line-height: 1.35;}
.tier b {display: block; font-size: 0.82rem;}
.tier.active {color: #ffffff;}
</style>
"""


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)


def _section(title, note=None):
    st.markdown(f"<p class='sec-title'>{title}</p>", unsafe_allow_html=True)
    if note:
        st.caption(note)


def _sub_label(text):
    st.markdown(f"<p class='sub-label'>{text}</p>", unsafe_allow_html=True)


def _ordinal(n):
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _check_grid(labels, cols, row, ncols):
    items = [(k, lab) for k, lab in labels.items() if k in cols]
    grid = st.columns(ncols)
    for i, (k, lab) in enumerate(items):
        row[k] = int(grid[i % ncols].checkbox(lab, key=k))


def _lab(container, label, default, lo, hi, key, step=1.0, fmt="%.1f"):
    # Most cosmetic patients have no preoperative labs, so "Not available" is the default.
    with container:
        na = st.session_state.get(f"na_{key}", True)
        v = st.number_input(label, min_value=float(lo), max_value=float(hi),
                            value=float(default), step=float(step), format=fmt,
                            key=key, disabled=na)
        na = st.checkbox("Not available", value=True, key=f"na_{key}")
    return np.nan if na else v


def sidebar_about():
    first = next(iter(appcfg.CHOSEN_MODEL_DICT))
    n_test = util.load_population_summary(first)["n_test"]
    with st.sidebar:
        st.subheader("About", anchor=False)
        st.markdown(
            "PRO-CABS estimates a woman's risk of complications in the 30 days after "
            "cosmetic breast surgery, abdominoplasty, or a combined operation, using "
            "information available before surgery.")
        st.markdown(
            "The models were built from ACS-NSQIP data for 2012–2021 and tested on "
            f"{n_test:,} later patients (2022–2024).")
        _sub_label("Reading the results")
        st.markdown(
            "- **Estimated risk**: calibrated probability of the outcome within 30 days.\n"
            "- **Risk tier**: Low (bottom half of predicted risk), Moderate, or High "
            "(top 10%), with the rate observed "
            "in that tier among 2022–2024 test patients.\n"
            "- **Percentile**: where the estimate falls among 2022–2024 test patients.\n"
            "- **Contributing factors**: how much each input moved this estimate.")
        with st.expander("Outcome definitions"):
            for key, label in OUTCOME_LABELS.items():
                st.markdown(f"**{label}.** {OUTCOME_DEFS[key]}")
        with st.expander("Who the models apply to"):
            st.markdown(
                "Adult women with a cosmetic indication (ICD-10 Z41.1 / ICD-9 V50.1) "
                "undergoing cosmetic breast surgery (augmentation, mastopexy, "
                "reduction, implant removal, capsule surgery), abdominoplasty, or a "
                "combined operation. Liposuction and umbilical hernia repair performed with these "
                "operations are included; other concurrent operations are not.")
        st.divider()
        st.caption("For research and clinical decision support only. Estimates do not "
                   "replace clinical judgment.")
        st.caption("Builds on the framework of Matar et al. (PRO-TONGUE). "
                   "[Source code](https://github.com/jj-leee/pro-cabs)")


def get_input_data():
    """Returns (input_df, has_procedure)."""
    pre = util.load_preprocessor()
    cols = list(pre.feature_names_in_)
    row = {}

    with st.container(border=True):
        _section("Patient", "The models were developed in women only.")
        c = st.columns(3)
        row["age"] = c[0].number_input("Age (years)", 18.0, 90.0, 40.0, 1.0,
                                       format="%.0f", key="age")
        row["race_new"] = c[1].selectbox("Race", util.category_options("race_new"), key="race")
        row["ethnicity_hispanic"] = util.yn(
            c[2].selectbox("Hispanic ethnicity", ["No", "Yes"], key="hisp"))
        c = st.columns(3)
        row["height"] = c[0].number_input("Height (in)", 48.0, 84.0, 64.0, 1.0,
                                          format="%.0f", key="ht")
        row["weight"] = c[1].number_input("Weight (lb)", 70.0, 450.0, 150.0, 1.0,
                                          format="%.0f", key="wt")
        bmi = 703.0 * row["weight"] / row["height"] ** 2
        c[2].markdown(f"<div class='calc-label'>BMI</div>"
                      f"<div class='calc-value'>{bmi:.1f} kg/m²</div>"
                      "<div class='calc-note'>Calculated from height and weight</div>",
                      unsafe_allow_html=True)

    with st.container(border=True):
        _section("Health status")
        c = st.columns(3)
        row["asaclas"] = util.ASA_MAP[
            c[0].selectbox("ASA class", list(util.ASA_MAP), index=1, key="asa")]
        row["diabetes"] = util.DIABETES_MAP[
            c[1].selectbox("Diabetes", list(util.DIABETES_MAP), key="dm")]
        row["fnstatus2"] = util.FNSTATUS_MAP[
            c[2].selectbox("Functional status", list(util.FNSTATUS_MAP), key="fn")]
        _sub_label("Comorbidities")
        _check_grid(COMORBID_LABELS, cols, row, ncols=4)

    with st.container(border=True):
        _section("Preoperative labs",
                 "Optional. Values marked not available are imputed by the model "
                 "pipeline, as they were for the many patients in the development "
                 "data who had no preoperative labs.")
        c = st.columns(3)
        row["pralbum"] = _lab(c[0], "Albumin (g/dL)", 4.2, 1.0, 6.0, "alb", 0.1)
        row["prwbc"] = _lab(c[1], "WBC (×10³/µL)", 7.0, 1.0, 50.0, "wbc", 0.1)
        row["prhct"] = _lab(c[2], "Hematocrit (%)", 40.0, 15.0, 60.0, "hct", 0.1)
        c = st.columns(3)
        row["prplate"] = _lab(c[0], "Platelets (×10³/µL)", 250, 50, 1000, "plt", 1.0, "%.0f")
        row["prcreat"] = _lab(c[1], "Creatinine (mg/dL)", 0.8, 0.1, 14.0, "cr", 0.1)
        row["prsodm"] = _lab(c[2], "Sodium (mmol/L)", 139, 120, 155, "na", 1.0, "%.0f")

    with st.container(border=True):
        _section("Operation",
                 "Select every procedure planned for this operation. They are entered "
                 "as patient characteristics; the tool does not compare risk with "
                 "versus without a procedure.")
        _sub_label("Abdomen")
        _check_grid(ABDO_LABELS, cols, row, ncols=3)
        _sub_label("Breast")
        _check_grid(BREAST_LABELS, cols, row, ncols=3)
        _sub_label("Performed with the operation")
        _check_grid(ADJUNCT_LABELS, cols, row, ncols=3)
        any_breast = any(row.get(k, 0) for k in BREAST_LABELS)
        row["combined_procedure"] = int(bool(row.get("abdominoplasty", 0)) and any_breast)
        c = st.columns(3)
        row["acuity"] = c[0].selectbox("Acuity", util.category_options("acuity"),
                                       format_func=lambda s: s.capitalize(), key="acu")
        row["anesthes"] = c[1].selectbox("Anesthesia", util.category_options("anesthes"),
                                         key="an")
        row["surgspec"] = c[2].selectbox("Surgical specialty",
                                         util.category_options("surgspec"), key="ss")

    # fill any pipeline column not collected (defensive)
    for c in cols:
        row.setdefault(c, np.nan if c in srccfg.NUMERIC_COLS else 0)
    input_df = pd.DataFrame({c: [row[c]] for c in cols})
    has_procedure = bool(row.get("abdominoplasty", 0)) or any_breast
    return input_df, has_procedure


def select_outcomes():
    """Outcome picker shown above the estimate button. Returns [(label, key)]."""
    keys = [k for k in OUTCOME_LABELS if k in appcfg.OUTCOMES.values()]
    _section("Outcomes")
    chosen = st.pills("Outcomes", keys, selection_mode="multi", default=keys,
                      format_func=OUTCOME_LABELS.get, key="outcomes",
                      label_visibility="collapsed")
    return [(OUTCOME_LABELS[k], k) for k in keys if k in (chosen or [])]


def predict(outcome, input_df):
    pre = util.load_preprocessor()
    feats = util.load_reduced_features()
    Xt = remove_prefix(pd.DataFrame(np.array(pre.transform(input_df)),
                                    columns=pre.get_feature_names_out()))
    for c in Xt.columns:
        Xt[c] = pd.to_numeric(Xt[c], errors="coerce")
    Xr = Xt[feats]
    model = util.load_model(outcome)
    return float(model.predict_proba(Xr)[0, 1]), Xr


def _result(label, outcome, input_df):
    prob, Xr = predict(outcome, input_df)
    cat, _, _, idx = util.get_risk_category(prob, outcome)
    thr = util.load_bin_thresholds(outcome)
    rates = util.bin_occur_rates(outcome, thr)
    pctile, n_pop = util.percentile_vs_population(prob, outcome)
    return dict(label=label, outcome=outcome, prob=prob, Xr=Xr, cat=cat, idx=idx,
                thr=thr, rates=rates, pctile=pctile, n_pop=n_pop)


def _tier_scale(r):
    t = [100 * float(x) for x in r["thr"]]
    ranges = ([f"&lt; {t[0]:.1f}%"]
              + [f"{t[i]:.1f}–{t[i + 1]:.1f}%" for i in range(len(t) - 1)]
              + [f"≥ {t[-1]:.1f}%"])
    cells = []
    for i, (name, rng) in enumerate(zip(util.RISK_LABELS, ranges)):
        obs = r["rates"][i]
        obs_txt = "—" if obs is None or np.isnan(obs) else f"{obs:.1%}"
        active = i == r["idx"]
        style = f" style='background:{TIER_COLORS[i]}'" if active else ""
        cells.append(f"<div class='tier{' active' if active else ''}'{style}>"
                     f"<b>{name}</b>{rng}<br>Observed {obs_txt}</div>")
    return f"<div class='tier-scale'>{''.join(cells)}</div>"


def show_results(selected, input_df):
    results = [_result(label, outcome, input_df) for label, outcome in selected]

    st.subheader("Estimated 30-day risk", anchor=False)
    cards = st.columns(len(results))
    for col, r in zip(cards, results):
        with col.container(border=True):
            st.markdown(f"<div class='card-label'>{r['label']}</div>"
                        f"<div class='risk-value'>{r['prob']:.1%}</div>",
                        unsafe_allow_html=True)
            st.badge(f"{r['cat']} risk", color=TIER_BADGES[r["idx"]])
            st.caption(f"{_ordinal(r['pctile'])} percentile")

    tabs = st.tabs([r["label"] for r in results])
    for tab, r in zip(tabs, results):
        with tab:
            left, right = st.columns([2, 3], gap="large")
            with left:
                st.markdown(f"Estimated risk **{r['prob']:.1%}**, in the "
                            f"**{r['cat']}** risk tier.")
                st.markdown(_tier_scale(r), unsafe_allow_html=True)
                obs = r["rates"][r["idx"]]
                if obs is not None and not np.isnan(obs):
                    st.markdown(f"In the 2022–2024 test cohort (n = {r['n_pop']:,}), "
                                f"**{obs:.1%}** of patients in the {r['cat']} tier "
                                "had this outcome.")
                st.markdown(f"This estimate is higher than **{r['pctile']:.0f}%** of "
                            "estimates for 2022–2024 test patients.")
                model = MODEL_NAMES.get(appcfg.CHOSEN_MODEL_DICT[r["outcome"]], "")
                st.caption(f"**Definition.** {OUTCOME_DEFS[r['outcome']]}  \n"
                           f"**Model.** Calibrated {model}.")
            with right:
                _section("Contributing factors")
                try:
                    expl = util.load_explainer(r["outcome"])
                    names, phi = appshap.compute_shap(
                        expl, r["Xr"], key=f"{r['outcome']}_{hash(r['Xr'].to_json())}")
                    st.altair_chart(appshap.shap_chart(names, phi, 10), width="stretch")
                    st.caption("Each bar is a factor's share of the total contribution "
                               "to this estimate. Red factors raise the risk; blue "
                               "factors lower it.")
                except Exception as e:
                    st.caption(f"Contributing factors are unavailable: {e}")
