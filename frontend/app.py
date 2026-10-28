import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Add project root directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.data_loader import get_fd003_data
from backend.investigate import investigate_engine, load_all_models
from backend.report_generator import generate_json_report, generate_text_report

# Page Config
st.set_page_config(
    page_title="Turbofan Engine Failure Investigator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .status-badge-failed {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-badge-healthy {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_dataset():
    """Loads and caches test dataset."""
    train_df, test_df, rul_df, dropped_cols, source_used = get_fd003_data()
    return train_df, test_df, rul_df, dropped_cols, source_used


@st.cache_resource
def get_cached_models():
    """Loads and caches trained backend models."""
    return load_all_models()


def main():
    st.markdown('<div class="main-header">⚙️ Industrial Machine Failure Investigation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">NASA C-MAPSS Turbofan Degradation (FD003) • Anomaly Detection, State Mapping & Root-Cause Diagnosis</div>', unsafe_allow_html=True)

    # Load data and models
    try:
        train_df, test_df, rul_df, dropped_cols, source_used = get_cached_dataset()
        _ = get_cached_models()
    except Exception as e:
        st.error(f"Failed to load system data or models: {e}")
        st.info("Ensure you have run `python backend/train.py` to train and save model files.")
        return

    # Sidebar setup
    st.sidebar.header("Investigation Controls")
    test_engine_ids = sorted(test_df["engine_id"].unique())

    selected_engine_id = st.sidebar.selectbox(
        "Select Test Engine ID",
        options=test_engine_ids,
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Dataset Source:**\n`{source_used}`")
    st.sidebar.markdown(f"**Dropped Features (Zero Var):**\n`{', '.join(dropped_cols)}`")

    # Run Investigation
    try:
        report = investigate_engine(selected_engine_id, n_cycles_earlier=20)
    except Exception as e:
        st.error(f"Error during engine investigation: {e}")
        return

    engine_test_df = test_df[test_df["engine_id"] == selected_engine_id].sort_values("cycle")
    timeline_df = pd.DataFrame(report["timeline"])

    # KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Target Asset", f"Engine #{selected_engine_id}")
    with col2:
        st.metric("Total Cycles Logged", f"{len(engine_test_df)} cycles")
    with col3:
        status_html = '<span class="status-badge-failed">FAILURE DETECTED</span>' if report["failure_detected"] else '<span class="status-badge-healthy">NORMAL STATE</span>'
        st.markdown(f"**Health Status**\n\n{status_html}", unsafe_allow_html=True)
    with col4:
        st.metric("Diagnosed Failure Type", report["failure_type"], f"{report['confidence']*100:.1f}% confidence")

    st.markdown("---")

    # 1. Line Chart of Raw Key Sensor Readings
    st.subheader("1. Raw Key Sensor Readings Over Cycles")
    available_sensors = [c for c in engine_test_df.columns if c.startswith("sensor")]
    default_sensors = [s for s in ["sensor2", "sensor3", "sensor4", "sensor7", "sensor12", "sensor15"] if s in available_sensors]

    selected_sensors = st.multiselect(
        "Select Sensor Readings to Display:",
        options=available_sensors,
        default=default_sensors if default_sensors else available_sensors[:4]
    )

    if selected_sensors:
        fig1, ax1 = plt.subplots(figsize=(10, 3.5))
        for sensor in selected_sensors:
            ax1.plot(engine_test_df["cycle"], engine_test_df[sensor], label=sensor, linewidth=1.5)
        ax1.set_xlabel("Operational Cycle")
        ax1.set_ylabel("Sensor Reading")
        ax1.set_title(f"Engine #{selected_engine_id} Raw Sensor Trajectories")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        plt.tight_layout()
        st.pyplot(fig1)

    # 2. VAE Anomaly Score Chart
    st.subheader("2. VAE Reconstruction Anomaly Score")
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.plot(timeline_df["cycle"], timeline_df["anomaly_score"], color="#7C3AED", label="VAE Anomaly Score", linewidth=2)
    threshold = report.get("vae_threshold", 0.05)
    ax2.axhline(threshold, color="#EF4444", linestyle="--", label=f"Anomaly Threshold ({threshold:.4f})", linewidth=1.5)
    ax2.fill_between(timeline_df["cycle"], timeline_df["anomaly_score"], threshold, where=(timeline_df["anomaly_score"] > threshold), color="#FCA5A5", alpha=0.5)
    ax2.set_xlabel("Operational Cycle")
    ax2.set_ylabel("Reconstruction Error (MSE)")
    ax2.set_title("Health Degradation Anomaly Trajectory")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig2)

    # 3. Health State Timeline
    st.subheader("3. Discrete Health-State Timeline")
    state_numeric_map = {"Normal": 0, "Degraded": 1, "Critical": 2, "Failure": 3}
    timeline_df["state_num"] = timeline_df["state"].map(state_numeric_map)

    fig3, ax3 = plt.subplots(figsize=(10, 2.5))
    ax3.step(timeline_df["cycle"], timeline_df["state_num"], where="post", color="#2563EB", linewidth=2.5)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(["Normal", "Degraded", "Critical", "Failure"])
    ax3.set_xlabel("Operational Cycle")
    ax3.set_title("Health State Degradation Progression")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Detect state transition cycles
    transitions = []
    curr_state = None
    for _, row in timeline_df.iterrows():
        if row["state"] != curr_state:
            curr_state = row["state"]
            transitions.append((row["cycle"], row["state"]))

    for cyc, st_name in transitions:
        if st_name != "Normal":
            ax3.annotate(f"{st_name} (Cycle {cyc})", xy=(cyc, state_numeric_map[st_name]),
                         xytext=(cyc, state_numeric_map[st_name] + 0.3),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                         fontsize=9, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig3)

    # Display transition event list
    with st.expander("View State Transition Events"):
        for cyc, st_name in transitions:
            st.write(f"- **Cycle {cyc}**: Transitioned to state **{st_name}**")

    # 4. Root Cause Section
    st.subheader("4. Root Cause & Bayesian Diagnostic Evidence")
    r_col1, r_col2 = st.columns([1, 1])

    with r_col1:
        st.markdown(f"**Predicted Root Cause:** `{report['root_cause']}`")
        st.markdown(f"**Confidence Level:** `{report['confidence']*100:.2f}%`")
        st.markdown("**Affected Components:**")
        for comp in report["affected_components"]:
            st.markdown(f"- {comp}")

    with r_col2:
        st.markdown("**Bayesian Network Evidence:**")
        for ev in report["evidence"]:
            st.markdown(f"- {ev}")

    # 5. Counterfactual What-If Panel
    st.subheader("5. Counterfactual 'What-If' Maintenance Simulation")
    st.markdown("Simulate how earlier preventive intervention would impact asset RUL.")

    max_c = len(engine_test_df)
    n_cycles = st.slider(
        "Select number of cycles earlier for simulated maintenance:",
        min_value=5,
        max_value=min(60, max_c - 5),
        value=20,
        step=5
    )

    if st.button("Simulate Early Maintenance Outcome"):
        sim_report = investigate_engine(selected_engine_id, n_cycles_earlier=n_cycles)
        what_if_res = sim_report["what_if"]

        st.info(f"**Scenario:** {what_if_res['scenario']}")
        st.success(f"**Simulated Outcome:** {what_if_res['result']}")

    # 6. Report Download Section
    st.subheader("6. Download Full Investigation Report")

    json_str = generate_json_report(report)
    text_str = generate_text_report(report)

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.download_button(
            label="📄 Download Full JSON Report",
            data=json_str,
            file_name=f"engine_{selected_engine_id}_investigation.json",
            mime="application/json"
        )
    with d_col2:
        st.download_button(
            label="📝 Download Human-Readable Text Report",
            data=text_str,
            file_name=f"engine_{selected_engine_id}_investigation.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()
