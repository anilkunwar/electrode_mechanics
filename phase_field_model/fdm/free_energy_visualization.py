import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit interface
st.title("Free Energy Visualization for Sn to Li₂Sn₅ Phase Transformation")
st.markdown(
    """
    Visualize the chemical, electromigration, and elastic free energy densities (J/m³) as functions of Sn mole fraction \(c\) and order parameter \(\eta_1\) (Sn phase).
    Energies are converted from J/mol using Sn molar volume (16.29e-6 m³/mol). Electromigration is scaled to match chemical energy magnitude.
    Elastic energy includes a phase-dependent term to ensure negative minima, consistent with chemical and electromigration energies.
    Adjust \(\phi\) (negative for lithiation) to observe equilibrium (\(\phi = 0\)) and Li₂Sn₅ preference (\(\phi < 0\)).
    """
)

# Parameters
st.sidebar.header("Simulation Parameters")
V_m_Sn = 16.29e-6  # Molar volume (m³/mol)
W_mol = 1.629  # J/mol
W = W_mol / V_m_Sn  # 1e5 J/m³
G_Sn_mol = -0.402363  # J/mol
G_Li2Sn5_mol = -0.53757  # J/mol
E_Sn_mol = -0.4887  # J/mol
E_Li2Sn5_mol = -0.6501  # J/mol
G_Sn = G_Sn_mol / V_m_Sn  # -2.47e4 J/m³
G_Li2Sn5 = G_Li2Sn5_mol / V_m_Sn  # -3.3e4 J/m³
E_Sn = E_Sn_mol / V_m_Sn  # -3e4 J/m³
E_Li2Sn5 = E_Li2Sn5_mol / V_m_Sn  # -3.99e4 J/m³
c_Sn = 0.95
c_Li2Sn5 = 5/7
epsilon = 1e-6
N_A = 6.022e23
e = 1.602e-19
Z_Sn = 2
Z_Li2Sn5 = 30
R = 8.314
T = 523.15
F = 96485
phi_0 = R * T / F
phi = st.sidebar.slider("Electric Potential φ (V)", -0.1, 0.0, 0.0, step=0.01, format="%.3f")
gamma = st.sidebar.slider("Electromigration Scaling Factor (γ)", 0.0, 2000.0, 1000.0, step=100.0)
alpha = st.sidebar.slider("Eigenstrain Scaling Factor (α)", 0.0, 1.0, 0.1, step=0.01)
E_Sn = st.sidebar.number_input("Sn Young's Modulus (Pa)", min_value=1e5, max_value=1e7, value=1e6, step=1e5, format="%.1e")
nu_Sn = 0.33
E_Li2Sn5 = st.sidebar.number_input("Li₂Sn₅ Young's Modulus (Pa)", min_value=1e5, max_value=1e7, value=1e6, step=1e5, format="%.1e")
nu_Li2Sn5 = 0.3
Vm_Li2Sn5 = 20.135e-6
beta_2D = 0.5 * np.log(Vm_Li2Sn5 / V_m_Sn) / (1 - c_Li2Sn5)

# Elastic constants
C_11_Sn = C_22_Sn = E_Sn / (1 - nu_Sn**2)
C_12_Sn = C_21_Sn = E_Sn * nu_Sn / (1 - nu_Sn**2)
C_11_Li2Sn5 = C_22_Li2Sn5 = E_Li2Sn5 / (1 - nu_Li2Sn5**2)
C_12_Li2Sn5 = C_21_Li2Sn5 = E_Li2Sn5 * nu_Li2Sn5 / (1 - nu_Li2Sn5**2)

# Colormap selection
colormap_options = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Hot', 'Jet', 'Rainbow', 'RdBu']
colormaps = {
    "Chemical": st.sidebar.selectbox("Chemical Free Energy Colormap", colormap_options, index=colormap_options.index("Viridis")),
    "Electromigration": st.sidebar.selectbox("Electromigration Free Energy Colormap", colormap_options, index=colormap_options.index("RdBu")),
    "Elastic": st.sidebar.selectbox("Elastic Free Energy Colormap", colormap_options, index=colormap_options.index("Plasma")),
    "Total": st.sidebar.selectbox("Total Free Energy Colormap", colormap_options, index=colormap_options.index("Jet")),
}

# Free energy functions
def h(eta_i, eta1, eta2, eta3):
    eta_sum = eta1**2 + eta2**2 + eta3**2 + epsilon
    return eta_i**2 / eta_sum

def f_chemical(c, eta1, eta2, eta3):
    h1 = h(eta1, eta1, eta2, eta3)
    h2 = h(eta2, eta1, eta2, eta3)
    h3 = h(eta3, eta1, eta2, eta3)
    g = 6 * (eta1**2 * (1 - eta1)**2 + eta2**2 * (1 - eta2)**2 + eta3**2 * (1 - eta3)**2)
    return (h1 * G_Sn_mol + (h2 + h3) * G_Li2Sn5_mol) * c + W_mol * (c - c_Sn) * (c - c_Li2Sn5) + W_mol * g

def f_electromigration(c, eta1, eta2, eta3, phi):
    h1 = h(eta1, eta1, eta2, eta3)
    h2 = h(eta2, eta1, eta2, eta3)
    h3 = h(eta3, eta1, eta2, eta3)
    return -gamma * (N_A * e / V_m_Sn) * ((h2 + h3) * Z_Li2Sn5 + h1 * Z_Sn) * (1 - c) * (abs(phi) / phi_0)

def f_elastic(c, eta1, eta2, eta3):
    h1 = h(eta1, eta1, eta2, eta3)
    h2 = h(eta2, eta1, eta2, eta3)
    h3 = h(eta3, eta1, eta2, eta3)
    C_11 = (h2 + h3) * C_11_Li2Sn5 + h1 * C_11_Sn
    C_12 = (h2 + h3) * C_12_Li2Sn5 + h1 * C_12_Sn
    C_21 = (h2 + h3) * C_21_Li2Sn5 + h1 * C_21_Sn
    C_22 = (h2 + h3) * C_22_Li2Sn5 + h1 * C_22_Sn
    epsilon_xx = alpha * beta_2D * (1 - c) * (h2 + h3)
    return ((0.5 * (C_11 + C_12 + C_21 + C_22) * epsilon_xx**2) / V_m_Sn + (h1 * E_Sn + (h2 + h3) * E_Li2Sn5) * c)

# Compute free energies
c = np.linspace(0, 1, 100)
eta1 = np.linspace(0, 1, 100)
C, ETA1 = np.meshgrid(c, eta1)
ETA2 = 1 - ETA1
ETA3 = np.zeros_like(ETA1)
F_CHEM = f_chemical(C, ETA1, ETA2, ETA3) / V_m_Sn
F_ELEC = f_electromigration(C, ETA1, ETA2, ETA3, phi)
F_ELASTIC = f_elastic(C, ETA1, ETA2, ETA3)
F_TOTAL = F_CHEM + F_ELEC + F_ELASTIC

# Find minima and maxima
def find_peaks(F, c, eta1):
    min_idx = np.unravel_index(np.argmin(F), F.shape)
    max_idx = np.unravel_index(np.argmax(F), F.shape)
    return {
        "min_value": F[min_idx], "min_c": c[min_idx[1]], "min_eta1": eta1[min_idx[0]],
        "max_value": F[max_idx], "max_c": c[max_idx[1]], "max_eta1": eta1[max_idx[0]]
    }

peaks = {
    "Chemical": find_peaks(F_CHEM, c, eta1),
    "Electromigration": find_peaks(F_ELEC, c, eta1),
    "Elastic": find_peaks(F_ELASTIC, c, eta1),
    "Total": find_peaks(F_TOTAL, c, eta1)
}

# Evaluate at equilibrium points
f_chem_sn = f_chemical(0.95, 1, 0, 0) / V_m_Sn
f_elec_sn = f_electromigration(0.95, 1, 0, 0, phi)
f_elastic_sn = f_elastic(0.95, 1, 0, 0)
f_total_sn = f_chem_sn + f_elec_sn + f_elastic_sn
f_chem_li2sn5 = f_chemical(0.714, 0, 1, 0) / V_m_Sn
f_elec_li2sn5 = f_electromigration(0.714, 0, 1, 0, phi)
f_elastic_li2sn5 = f_elastic(0.714, 0, 1, 0)
f_total_li2sn5 = f_chem_li2sn5 + f_elec_li2sn5 + f_elastic_li2sn5

# Plotting
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}],
           [{"type": "surface"}, {"type": "contour"}]],
    subplot_titles=("Chemical Free Energy", "Electromigration Free Energy", "Elastic Free Energy", "Total Free Energy (Contour)")
)

# Chemical free energy surface
fig.add_trace(
    go.Surface(x=c, y=eta1, z=F_CHEM, colorscale=colormaps["Chemical"], showscale=True),
    row=1, col=1
)
fig.add_trace(
    go.Scatter3d(x=[0.95, 0.714], y=[1, 0], z=[f_chem_sn, f_chem_li2sn5], mode="markers", marker=dict(size=5, color="red")),
    row=1, col=1
)

# Electromigration free energy surface
fig.add_trace(
    go.Surface(x=c, y=eta1, z=F_ELEC, colorscale=colormaps["Electromigration"], showscale=True),
    row=1, col=2
)
fig.add_trace(
    go.Scatter3d(x=[0.95, 0.714], y=[1, 0], z=[f_elec_sn, f_elec_li2sn5], mode="markers", marker=dict(size=5, color="red")),
    row=1, col=2
)

# Elastic free energy surface
fig.add_trace(
    go.Surface(x=c, y=eta1, z=F_ELASTIC, colorscale=colormaps["Elastic"], showscale=True),
    row=2, col=1
)
fig.add_trace(
    go.Scatter3d(x=[0.95, 0.714], y=[1, 0], z=[f_elastic_sn, f_elastic_li2sn5], mode="markers", marker=dict(size=5, color="red")),
    row=2, col=1
)

# Total free energy contour
fig.add_trace(
    go.Contour(x=c, y=eta1, z=F_TOTAL, colorscale=colormaps["Total"], showscale=True),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=[0.95, 0.714], y=[1, 0], mode="markers", marker=dict(size=10, color="red", symbol="circle")),
    row=2, col=2
)

# Update layout
fig.update_layout(
    height=800,
    showlegend=False,
    scene=dict(xaxis_title="c (Sn Mole Fraction)", yaxis_title="η₁ (Sn Phase)", zaxis_title="Energy (J/m³)"),
    scene2=dict(xaxis_title="c", yaxis_title="η₁", zaxis_title="Energy (J/m³)"),
    scene3=dict(xaxis_title="c", yaxis_title="η₁", zaxis_title="Energy (J/m³)")
)
fig.update_scenes(aspectmode="cube")

# Display plot
st.plotly_chart(fig, use_container_width=True)

# Display peaks
st.header("Minima and Maxima (J/m³)")
st.write("Values at equilibrium points (Sn: c = 0.95, η₁ = 1; Li₂Sn₅: c = 0.714, η₂ = 1) and grid maxima:")
st.markdown(
    f"""
    | Energy Type | Sn (c = 0.95, η₁ = 1) | Li₂Sn₅ (c = 0.714, η₂ = 1) | Grid Minimum (c, η₁) | Grid Maximum (c, η₁) |
    |-------------|-----------------------|-----------------------------|----------------------|----------------------|
    | Chemical    | {f_chem_sn:.2e} | {f_chem_li2sn5:.2e} | {peaks['Chemical']['min_value']:.2e} ({peaks['Chemical']['min_c']:.3f}, {peaks['Chemical']['min_eta1']:.3f}) | {peaks['Chemical']['max_value']:.2e} ({peaks['Chemical']['max_c']:.3f}, {peaks['Chemical']['max_eta1']:.3f}) |
    | Electromigration | {f_elec_sn:.2e} | {f_elec_li2sn5:.2e} | {peaks['Electromigration']['min_value']:.2e} ({peaks['Electromigration']['min_c']:.3f}, {peaks['Electromigration']['min_eta1']:.3f}) | {peaks['Electromigration']['max_value']:.2e} ({peaks['Electromigration']['max_c']:.3f}, {peaks['Electromigration']['max_eta1']:.3f}) |
    | Elastic     | {f_elastic_sn:.2e} | {f_elastic_li2sn5:.2e} | {peaks['Elastic']['min_value']:.2e} ({peaks['Elastic']['min_c']:.3f}, {peaks['Elastic']['min_eta1']:.3f}) | {peaks['Elastic']['max_value']:.2e} ({peaks['Elastic']['max_c']:.3f}, {peaks['Elastic']['max_eta1']:.3f}) |
    | Total       | {f_total_sn:.2e} | {f_total_li2sn5:.2e} | {peaks['Total']['min_value']:.2e} ({peaks['Total']['min_c']:.3f}, {peaks['Total']['min_eta1']:.3f}) | {peaks['Total']['max_value']:.2e} ({peaks['Total']['max_c']:.3f}, {peaks['Total']['max_eta1']:.3f}) |
    """
)

# Equilibrium check
st.header("Equilibrium Analysis")
if phi == 0:
    st.write("At φ = 0, no electromigration. Check equilibrium between Sn (c = 0.95, η₁ = 1) and Li₂Sn₅ (c = 0.714, η₂ = 1):")
    st.write(f"Sn Total Free Energy: {f_total_sn:.2e} J/m³")
    st.write(f"Li₂Sn₅ Total Free Energy: {f_total_li2sn5:.2e} J/m³")
    if abs(f_total_sn - f_total_li2sn5) < 1e-2 * W:
        st.success("Sn and Li₂Sn₅ are in equilibrium (similar free energies).")
    else:
        st.warning("Sn and Li₂Sn₅ free energies differ significantly.")
else:
    st.write(f"At φ = {phi:.3f} V, electromigration favors Li₂Sn₅:")
    st.write(f"Sn Total Free Energy: {f_total_sn:.2e} J/m³")
    st.write(f"Li₂Sn₅ Total Free Energy: {f_total_li2sn5:.2e} J/m³")
    mu_sn = ((G_Sn_mol + W_mol * (2 * 0.95 - c_Sn - c_Li2Sn5)) / V_m_Sn - gamma * (N_A * e / V_m_Sn) * Z_Sn * (abs(phi) / phi_0) + E_Sn) * V_m_Sn / V_m_Sn
    mu_li2sn5 = ((G_Li2Sn5_mol + W_mol * (2 * 0.714 - c_Sn - c_Li2Sn5)) / V_m_Sn - gamma * (N_A * e / V_m_Sn) * Z_Li2Sn5 * (abs(phi) / phi_0) - alpha * beta_2D * (C_11_Li2Sn5 + C_12_Li2Sn5 + C_21_Li2Sn5 + C_22_Li2Sn5) * (alpha * beta_2D * (1 - 0.714)) + E_Li2Sn5) * V_m_Sn / V_m_Sn
    st.write(f"Chemical Potential at Sn: {mu_sn:.2e} J/m³")
    st.write(f"Chemical Potential at Li₂Sn₅: {mu_li2sn5:.2e} J/m³")
    if mu_li2sn5 < mu_sn:
        st.success("Li₂Sn₅ has lower chemical potential, favored during lithiation.")
    else:
        st.warning("Li₂Sn₅ not favored. Adjust parameters.")