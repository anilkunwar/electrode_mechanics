import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Streamlit interface
st.title("Eigenstrain Calculator for Sn to Li₂Sn₅ Transformation")
st.markdown(
    """
    This tool calculates the eigenstrain due to the volume change from BCT Sn to Li₂Sn₅ phase transformation using two methods:

    1. **Linear strain**: εₓₓ = α × (1/3) × (V<sub>Li₂Sn₅</sub> - V<sub>Sn</sub>) / V<sub>Sn</sub>  
    2. **Logarithmic strain**: εₓₓ = (1/2) × ln(V<sub>Li₂Sn₅</sub> / V<sub>Sn</sub>)

    You may also compute elastic stresses based on elastic constants.
    """,
    unsafe_allow_html=True
)

# Input parameters
st.sidebar.header("Input Parameters")
V_Sn = st.sidebar.number_input("Molar Volume of Sn (cm³/mol)", min_value=0.1, max_value=100.0, value=16.29, step=0.01)
V_Li2Sn5 = st.sidebar.number_input("Molar Volume of Li₂Sn₅ (cm³/mol)", min_value=0.1, max_value=100.0, value=20.135, step=0.01)
alpha = st.sidebar.number_input("Eigenstrain Scaling Factor α", min_value=0.0, max_value=1.0, value=1.0, step=0.01)

# Elastic constants (optional)
st.sidebar.header("Elastic Constants (Optional)")
compute_stress = st.sidebar.checkbox("Compute Elastic Stresses", value=False)
if compute_stress:
    E_Sn = st.sidebar.number_input("Sn Young's Modulus (GPa)", min_value=0.1, max_value=1000.0, value=50.0, step=1.0)
    nu_Sn = st.sidebar.number_input("Sn Poisson's Ratio", min_value=0.0, max_value=0.49, value=0.3, step=0.01)
    E_Li2Sn5 = st.sidebar.number_input("Li₂Sn₅ Young's Modulus (GPa)", min_value=0.1, max_value=1000.0, value=40.0, step=1.0)
    nu_Li2Sn5 = st.sidebar.number_input("Li₂Sn₅ Poisson's Ratio", min_value=0.0, max_value=0.49, value=0.25, step=0.01)

# Eigenstrain calculation
relative_volume_change = (V_Li2Sn5 - V_Sn) / V_Sn
volume_ratio = V_Li2Sn5 / V_Sn
eigenstrain_linear = alpha * (1/3) * relative_volume_change
eigenstrain_log = (1/2) * np.log(volume_ratio) if volume_ratio > 0 else 0.0

# Stress calculation (isotropic linear elastic)
if compute_stress:
    denom = (1 - nu_Li2Sn5**2)
    if denom == 0:
        st.error("Invalid Poisson's ratio (ν ≈ 1). Cannot compute stress.")
        stress_xx_Li2Sn5_linear = stress_xx_Li2Sn5_log = 0.0
    else:
        C_11 = E_Li2Sn5 / denom
        C_12 = E_Li2Sn5 * nu_Li2Sn5 / denom
        stress_xx_Li2Sn5_linear = (C_11 + C_12) * eigenstrain_linear
        stress_xx_Li2Sn5_log = (C_11 + C_12) * eigenstrain_log
else:
    stress_xx_Li2Sn5_linear = stress_xx_Li2Sn5_log = 0.0

# Display results
st.header("Results")
st.write(f"**Molar Volume of Sn**: {V_Sn:.2f} cm³/mol")
st.write(f"**Molar Volume of Li₂Sn₅**: {V_Li2Sn5:.2f} cm³/mol")
st.write(f"**Relative Volume Change**: {relative_volume_change * 100:.2f}%")
st.write(f"**Eigenstrain (Linear)** εₓₓ: {eigenstrain_linear:.6f}")
st.write(f"**Eigenstrain (Logarithmic)** εₓₓ: {eigenstrain_log:.6f}")
if compute_stress:
    st.write(f"**Stress in Li₂Sn₅ (Linear)** σₓₓ: {stress_xx_Li2Sn5_linear:.2f} GPa")
    st.write(f"**Stress in Li₂Sn₅ (Logarithmic)** σₓₓ: {stress_xx_Li2Sn5_log:.2f} GPa")

# Visualization
st.header("Visualization")

fig = go.Figure()

# Volume bars
fig.add_trace(go.Bar(
    x=["Sn", "Li₂Sn₅"],
    y=[V_Sn, V_Li2Sn5],
    name="Molar Volume (cm³/mol)",
    marker_color=["#1f77b4", "#ff7f0e"]
))

# Eigenstrain bars
fig.add_trace(go.Bar(
    x=["Eigenstrain (Linear)", "Eigenstrain (Logarithmic)"],
    y=[eigenstrain_linear, eigenstrain_log],
    name="Eigenstrain εₓₓ",
    marker_color=["#2ca02c", "#17becf"]
))

# Stress bars
if compute_stress:
    fig.add_trace(go.Bar(
        x=["Stress (Linear)", "Stress (Logarithmic)"],
        y=[stress_xx_Li2Sn5_linear, stress_xx_Li2Sn5_log],
        name="Stress σₓₓ (GPa)",
        marker_color=["#d62728", "#9467bd"]
    ))

fig.update_layout(
    title="Volume, Eigenstrain, and Stress Comparison",
    barmode="group",
    xaxis_title="Quantity",
    yaxis_title="Value",
    legend=dict(x=0.01, y=0.99),
    height=500
)
st.plotly_chart(fig)

# Gauge chart
st.subheader("Eigenstrain Gauges")
fig_gauge = go.Figure()

fig_gauge.add_trace(go.Indicator(
    mode="gauge+number",
    value=eigenstrain_linear,
    title={"text": "Linear Eigenstrain"},
    gauge={"axis": {"range": [-0.1, 0.1]}, "bar": {"color": "#2ca02c"}},
    domain={"row": 0, "column": 0}
))
fig_gauge.add_trace(go.Indicator(
    mode="gauge+number",
    value=eigenstrain_log,
    title={"text": "Logarithmic Eigenstrain"},
    gauge={"axis": {"range": [-0.1, 0.1]}, "bar": {"color": "#17becf"}},
    domain={"row": 0, "column": 1}
))

fig_gauge.update_layout(grid={"rows": 1, "columns": 2}, height=400)
st.plotly_chart(fig_gauge)

