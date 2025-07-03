import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Streamlit interface
st.title("Eigenstrain Calculator for Sn to Li₂Sn₅ Transformation")
st.markdown(
    """
    This tool calculates the eigenstrain due to the volume change from BCT Sn to Li₂Sn₅ phase transformation using three methods:

    1. **Linear strain**: εₓₓ⁰ = εᵧᵧ⁰= α × (1/3) × (V<sub>Li₂Sn₅</sub> - V<sub>Sn</sub>) / V<sub>Sn</sub>  
    2. **Logarithmic strain**: εₓₓ⁰ = εᵧᵧ⁰ = α × (1/2) × ln(V<sub>Li₂Sn₅</sub> / V<sub>Sn</sub>)  
    3. **Stoichiometric strain : εₓₓ⁰ = εᵧᵧ⁰ = ∛(V<sub>Li₂Sn₅</sub> / V<sub>Sn</sub>) - 1  

    Select a model from the dropdown to compute the eigenstrain and optionally calculate elastic stresses.
    """,
    unsafe_allow_html=True
)

# Input parameters
st.sidebar.header("Input Parameters")
V_Sn = st.sidebar.number_input("Molar Volume of Sn (cm³/mol)", min_value=0.1, max_value=100.0, value=16.29, step=0.01)
V_Li2Sn5 = st.sidebar.number_input("Molar Volume of Li₂Sn₅ (cm³/mol)", min_value=0.1, max_value=100.0, value=20.135, step=0.01)
alpha = st.sidebar.number_input("Eigenstrain Scaling Factor α (for Linear and Logarithmic)", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
model = st.sidebar.selectbox("Select Eigenstrain Model", ["Linear", "Logarithmic", "Stoichiometric"])

# Elastic constants (optional)
st.sidebar.header("Elastic Constants (Optional)")
compute_stress = st.sidebar.checkbox("Compute Elastic Stresses", value=False)
if compute_stress:
    E_Sn = st.sidebar.number_input("Sn Young's Modulus (GPa)", min_value=0.1, max_value=1000.0, value=50.0, step=1.0)
    nu_Sn = st.sidebar.number_input("Sn Poisson's Ratio", min_value=0.0, max_value=0.49, value=0.3, step=0.01)
    E_Li2Sn5 = st.sidebar.number_input("Li₂Sn₅ Young's Modulus (GPa)", min_value=0.1, max_value=1000.0, value=40.0, step=1.0)
    nu_Li2Sn5 = st.sidebar.number_input("Li₂Sn₅ Poisson's Ratio", min_value=0.0, max_value=0.49, value=0.25, step=0.01)

# Eigenstrain calculation
relative_volume_change = (V_Li2Sn5 - V_Sn) / V_Sn if V_Sn > 0 else 0.0
volume_ratio = V_Li2Sn5 / V_Sn if V_Sn > 0 else 1.0
eigenstrain_linear = alpha * (1/3) * relative_volume_change
eigenstrain_log = alpha * (1/2) * np.log(volume_ratio) if volume_ratio > 0 else 0.0
eigenstrain_stoich = (volume_ratio ** (1/3)) - 1 if volume_ratio > 0 else 0.0

# Select eigenstrain based on model
if model == "Linear":
    eigenstrain = eigenstrain_linear
    model_label = "Linear"
elif model == "Logarithmic":
    eigenstrain = eigenstrain_log
    model_label = "Logarithmic"
else:  # Stoichiometric
    eigenstrain = eigenstrain_stoich
    model_label = "Stoichiometric "

# Stress calculation (isotropic linear elastic)
if compute_stress:
    denom = (1 - nu_Li2Sn5**2)
    if denom == 0:
        st.error("Invalid Poisson's ratio (ν ≈ 1). Cannot compute stress.")
        stress_xx_Li2Sn5 = 0.0
    else:
        C_11 = E_Li2Sn5 / denom
        C_12 = E_Li2Sn5 * nu_Li2Sn5 / denom
        stress_xx_Li2Sn5 = (C_11 + C_12) * eigenstrain
else:
    stress_xx_Li2Sn5 = 0.0

# Display results
st.header("Results")
st.write(f"**Molar Volume of Sn**: {V_Sn:.2f} cm³/mol")
st.write(f"**Molar Volume of Li₂Sn₅**: {V_Li2Sn5:.2f} cm³/mol")
st.write(f"**Relative Volume Change**: {relative_volume_change * 100:.2f}%")
st.write(f"**Eigenstrain ({model_label})** εₓₓ: {eigenstrain:.6f}")
st.write(f"**Eigenstrain (Linear)** εₓₓ: {eigenstrain_linear:.6f}")
st.write(f"**Eigenstrain (Logarithmic)** εₓₓ: {eigenstrain_log:.6f}")
st.write(f"**Eigenstrain (Stoichiometric)** εₓₓ: {eigenstrain_stoich:.6f}")
if compute_stress:
    st.write(f"**Stress in Li₂Sn₅ ({model_label})** σₓₓ: {stress_xx_Li2Sn5:.2f} GPa")

# Visualization
st.header("Visualization")

# Bar plot
fig = go.Figure()
fig.add_trace(go.Bar(
    x=["Sn", "Li₂Sn₅"],
    y=[V_Sn, V_Li2Sn5],
    name="Molar Volume (cm³/mol)",
    marker_color=["#1f77b4", "#ff7f0e"]
))
fig.add_trace(go.Bar(
    x=["Linear", "Logarithmic", "Stoichiometric"],
    y=[eigenstrain_linear, eigenstrain_log, eigenstrain_stoich],
    name="Eigenstrain εₓₓ",
    marker_color=["#2ca02c", "#17becf", "#9467bd"],
    yaxis="y2"
))
if compute_stress:
    stress_values = [stress_xx_Li2Sn5 if model == "Linear" else 0.0,
                    stress_xx_Li2Sn5 if model == "Logarithmic" else 0.0,
                    stress_xx_Li2Sn5 if model == "Stoichiometric" else 0.0]
    fig.add_trace(go.Bar(
        x=["Stress (Linear)", "Stress (Logarithmic)", "Stress (Stoichiometric)"],
        y=stress_values,
        name="Stress σₓₓ (GPa)",
        marker_color=["#d62728", "#e377c2", "#7f7f7f"],
        yaxis="y2"
    ))
fig.update_layout(
    title="Volume, Eigenstrain, and Stress Comparison",
    xaxis=dict(title="Quantity"),
    yaxis=dict(
        title=dict(text="Molar Volume (cm³/mol)", font=dict(color="#1f77b4")),
        tickfont=dict(color="#1f77b4")
    ),
    yaxis2=dict(
        title=dict(text="Eigenstrain / Stress (GPa)", font=dict(color="#2ca02c")),
        tickfont=dict(color="#2ca02c"),
        overlaying="y",
        side="right"
    ),
    barmode="group",
    legend=dict(x=0.01, y=0.99),
    height=500
)
st.plotly_chart(fig)

# Gauge chart
st.subheader("Eigenstrain Gauge")
fig_gauge = go.Figure()
fig_gauge.add_trace(go.Indicator(
    mode="gauge+number",
    value=eigenstrain,
    title=dict(text=f"{model_label} Eigenstrain", font=dict(size=14)),
    gauge={
        "axis": {"range": [-0.1, 0.1]},
        "bar": {"color": "#2ca02c" if model == "Linear" else "#17becf" if model == "Logarithmic" else "#9467bd"},
        "steps": [
            {"range": [-0.1, 0], "color": "lightgray"},
            {"range": [0, 0.1], "color": "gray"}
        ],
        "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 0}
    }
))
fig_gauge.update_layout(height=400)
st.plotly_chart(fig_gauge)

