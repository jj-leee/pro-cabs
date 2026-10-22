"""PRO-CABS Streamlit risk calculator (adapted from the ML-Tongue-Pred app).

Inputs = the deployed feature set; the planned procedures (abdominoplasty, breast
procedure types, liposuction, umbilical hernia repair) are user-set checkboxes,
and the combined abdominoplasty + breast flag is derived from them. Per selected
outcome the app shows the calibrated 30-day risk, the log-spaced risk category,
the observed in-category rate from the 2022-2024 test set, the outcome-specific
percentile, and a per-patient SHAP explanation.

Only the four primary outcomes are deployed (any, surgical and medical
complication, unplanned readmission). There is deliberately NO
procedure toggle/compare control -- the procedures are descriptive inputs to a
single prediction, never an A/B switch.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st
import display_functions as display


def main():
    st.set_page_config(page_title="PRO-CABS", page_icon="🏥", layout="wide")
    display.apply_style()
    display.sidebar_about()

    st.title("PRO-CABS: Postoperative Risk Outcomes after Cosmetic Abdominoplasty "
             "and Breast Surgery", anchor=False)
    st.caption("Estimates 30-day complication risk after cosmetic breast surgery, "
               "abdominoplasty, or both, from preoperative patient and operative "
               "factors. For research and clinical decision support only.")

    input_df, has_procedure = display.get_input_data()

    with st.container(border=True):
        selected = display.select_outcomes()
        ready = bool(selected) and has_procedure
        if st.button("Estimate risk", type="primary", disabled=not ready):
            st.session_state["go"] = True
        if not has_procedure:
            st.caption("Select abdominoplasty and/or at least one breast procedure.")
        elif not selected:
            st.caption("Select at least one outcome.")
    if not ready:
        return
    if st.session_state.get("go"):
        st.divider()
        display.show_results(selected, input_df)


if __name__ == "__main__":
    main()
