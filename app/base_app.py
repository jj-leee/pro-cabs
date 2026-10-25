"""PRO-CABS Streamlit risk calculator (adapted from the ML-Tongue-Pred app).

Inputs = the deployed feature set; the planned procedures (abdominoplasty, breast
procedure types, liposuction, umbilical hernia repair) are user-set checkboxes,
and the combined abdominoplasty + breast flag is derived from them. Per selected
outcome the app shows the calibrated 30-day risk, the risk tier (Low / Moderate /
High: development-set risk percentiles 0-50 / 50-90 / 90-100),
the observed in-tier rate from the 2022-2024 test set, the outcome-specific
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


# Header styles live here (not in display_functions) so a cached module on
# Streamlit Cloud can never leave the title unstyled.
HEADER_CSS = """
<style>
.pc-header {margin: 0.2rem 0 1.4rem 0; padding-bottom: 1.1rem; border-bottom: 1px solid #e5e7eb;}
.pc-title {font-size: 3rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.05;
           color: #111827; margin: 0;}
.pc-title .pc-accent {color: #c21615;}
.pc-name {font-size: 1.3rem; font-weight: 400; color: #374151; margin: 0.45rem 0 0 0;
          line-height: 1.4;}
.pc-name u {font-weight: 700; color: #111827; text-decoration-color: #c21615;
            text-decoration-thickness: 2px; text-underline-offset: 0.16em;}
.pc-tagline {font-size: 0.95rem; color: #6b7280; margin: 0.55rem 0 0 0; line-height: 1.5;}
</style>
"""


def header_html():
    initial = lambda w: f"<u>{w[0]}</u>{w[1:]}"
    full_name = " ".join([initial("Postoperative"), initial("Risk"), "of", initial("Outcomes"),
                          "after", initial("Cosmetic"), initial("Abdominoplasty"), "and/or",
                          initial("Breast"), initial("Surgery")])
    return (HEADER_CSS
            + "<div class='pc-header'>"
            "<div class='pc-title'>PRO-<span class='pc-accent'>CABS</span></div>"
            f"<div class='pc-name'>{full_name}</div>"
            "<div class='pc-tagline'>Estimates the risk of complications in the 30 days after "
            "cosmetic breast surgery, abdominoplasty, or a combined operation. For research "
            "and clinical decision support only.</div></div>")


def main():
    st.set_page_config(page_title="PRO-CABS", page_icon="🏥", layout="wide")
    display.apply_style()
    display.sidebar_about()

    st.markdown(header_html(), unsafe_allow_html=True)

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
