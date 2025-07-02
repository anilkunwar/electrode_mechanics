import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyvista as pv
import h5py
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_output = []

class StreamlitHandler(logging.Handler):
    def emit(self, record):
        log_output.append(self.format(record))

logger.handlers = [StreamlitHandler()]
logger.setLevel(logging.INFO)

# Configure PyVista
pv.global_theme.show_scalar_bar = True
pv.global_theme.jupyter_backend = "trame"

# Streamlit interface
st.title("Multiphase-Field Simulation for Sn → Li₂Sn₅ Transformation")
st.markdown(
    """
    Simulate phase transformation from Sn (η₁) to two Li₂Sn₅ grains (η₂, η₃) in a 2D domain.
    Free energy density (J/m³) is computed from J/mol using Sn molar volume (16.29e-6 m³/mol).
    Elastic energy includes a phase-dependent term to ensure negative minima.
    Adjust parameters to control lithiation (φ > 0) and visualize concentration, order parameters, potential, strains, and stresses.
    """
)

# Colormap options (Plotly-compatible)
colormap_options = [
    'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues',
    'blugrn', 'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
    'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser',
    'gnbu', 'gray', 'greens', 'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
    'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'oxy',
    'peach', 'phase', 'picnic', 'pinkyl', 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu',
    'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy', 'rdpu',
    'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed', 'sunset', 'sunsetdark',
    'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'turbo',
    'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
]

# Parameters
st.sidebar.header("Simulation Parameters")
Lx = st.sidebar.slider("Domain size X (m)", 1e-6, 3e-6, 1.5e-6, step=1e-7, format="%.1e")
Ly = st.sidebar.slider("Domain size Y (m)", 1e-6, 3e-6, 1.5e-6, step=1e-7, format="%.1e")
Nx = st.sidebar.slider("Grid points X", 20, 100, 20)
Ny = st.sidebar.slider("Grid points Y", 20, 100, 20)
t_max_dim = st.sidebar.number_input("Simulation time (s)", min_value=1e-28, max_value=1e-16, value=1e-17, step=1e-28, format="%.1e")
num_t_eval = st.sidebar.slider("Number of output time points", 3, 10, 5)
beta_dim = st.sidebar.number_input("Gradient energy coefficient η (J/m)", min_value=1e-10, max_value=1e-8, value=1e-9, step=1e-10, format="%.1e")
L = st.sidebar.number_input("Kinetic coefficient L (m³/(J·s))", min_value=1e8, max_value=1e10, value=5e8, step=1e8, format="%.1e")
M_Li2Sn5_dim = st.sidebar.number_input("Mobility Li₂Sn₅ (m²/(J·s))", min_value=1e-12, max_value=1e-8, value=1e-12, step=1e-12, format="%.1e")
M_Sn_dim = st.sidebar.number_input("Mobility Sn (m²/(J·s))", min_value=1e-12, max_value=1e-8, value=1e-12, step=1e-12, format="%.1e")
phi_anode = st.sidebar.number_input("Anode Voltage (V)", min_value=0.0, max_value=0.1, value=0.04, step=0.01, format="%.3f")
gamma = st.sidebar.number_input("Electromigration Scaling Factor (γ)", min_value=0.0, max_value=10000.0, value=1000.0, step=100.0)
alpha_eigenstrain = st.sidebar.number_input("Eigenstrain Scaling Factor (α)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
delta = 35e-9  # Interface width
sigma = 0.5  # Interfacial energy (J/m²)
kappa = 3/4 * sigma * delta
m = 6 * sigma / delta
L_sim = 16/3 * m * M_Sn_dim / kappa

# Elastic parameters
st.sidebar.header("Elastic Parameters (at 523.15 K)")
material_model = st.sidebar.selectbox("Material Model", ["Isotropic", "Orthotropic", "Anisotropic"])
if material_model == "Isotropic":
    E_Sn = st.sidebar.number_input("Sn Young's Modulus (Pa)", min_value=1e8, max_value=1e10, value=1e9, step=1e8, format="%.1e")
    nu_Sn = st.sidebar.number_input("Sn Poisson's Ratio", min_value=0.0, max_value=0.49, value=0.33, step=0.05)
    E_Li2Sn5 = st.sidebar.number_input("Li₂Sn₅ Young's Modulus (Pa)", min_value=1e8, max_value=1e10, value=1e9, step=1e8, format="%.1e")
    nu_Li2Sn5 = st.sidebar.number_input("Li₂Sn₅ Poisson's Ratio", min_value=0.0, max_value=0.49, value=0.3, step=0.05)
    C_11_Sn = C_22_Sn = E_Sn / (1 - nu_Sn**2)
    C_12_Sn = C_21_Sn = E_Sn * nu_Sn / (1 - nu_Sn**2)
    C_11_Li2Sn5 = C_22_Li2Sn5 = E_Li2Sn5 / (1 - nu_Li2Sn5**2)
    C_12_Li2Sn5 = C_21_Li2Sn5 = E_Li2Sn5 * nu_Li2Sn5 / (1 - nu_Li2Sn5**2)
else:
    C_11_Sn = st.sidebar.number_input("Sn C_11 (Pa)", min_value=1e8, max_value=1e10, value=1e9, step=1e8, format="%.1e")
    C_12_Sn = st.sidebar.number_input("Sn C_12 (Pa)", min_value=0.0, max_value=1e10, value=0.3e9, step=1e8, format="%.1e")
    C_21_Sn = C_12_Sn if material_model == "Orthotropic" else st.sidebar.number_input("Sn C_21 (Pa)", min_value=0.0, max_value=1e10, value=0.3e9, step=1e8, format="%.1e")
    C_22_Sn = st.sidebar.number_input("Sn C_22 (Pa)", min_value=1e8, max_value=1e10, value=1e9, step=1e8, format="%.1e")
    C_11_Li2Sn5 = st.sidebar.number_input("Li₂Sn₅ C_11 (Pa)", min_value=1e8, max_value=1e10, value=1e9, step=1e8, format="%.1e")
    C_12_Li2Sn5 = st.sidebar.number_input("Li₂Sn₅ C_12 (Pa)", min_value=0.0, max_value=1e10, value=0.3e9, step=1e8, format="%.1e")
    C_21_Li2Sn5 = C_12_Li2Sn5 if material_model == "Orthotropic" else st.sidebar.number_input("Li₂Sn₅ C_21 (Pa)", min_value=0.0, max_value=1e10, value=0.3e9, step=1e8, format="%.1e")
    C_22_Li2Sn5 = st.sidebar.number_input("Li₂Sn₅ C_22 (Pa)", min_value=1e8, max_value=1e10, value=1e9, step=1e8, format="%.1e")

# Colormap selection
st.sidebar.header("Visualization Settings")
colormaps = {
    "Concentration": st.sidebar.selectbox("Concentration Colormap", colormap_options, index=colormap_options.index("viridis")),
    "Eta_1": st.sidebar.selectbox("Eta_1 Colormap", colormap_options, index=colormap_options.index("jet")),
    "Eta_2": st.sidebar.selectbox("Eta_2 Colormap", colormap_options, index=colormap_options.index("rainbow")),
    "Eta_3": st.sidebar.selectbox("Eta_3 Colormap", colormap_options, index=colormap_options.index("hot")),
    "Potential": st.sidebar.selectbox("Potential Colormap", colormap_options, index=colormap_options.index("rdbu")),
    "Strain_xx": st.sidebar.selectbox("Strain_xx Colormap", colormap_options, index=colormap_options.index("rdbu")),
    "Stress_xx": st.sidebar.selectbox("Stress_xx Colormap", colormap_options, index=colormap_options.index("plasma")),
    "Stress_yy": st.sidebar.selectbox("Stress_yy Colormap", colormap_options, index=colormap_options.index("magma")),
    "Von_Mises_Stress": st.sidebar.selectbox("Von Mises Stress Colormap", colormap_options, index=colormap_options.index("rdbu")),
    "Chemical": st.sidebar.selectbox("Chemical Free Energy Colormap", colormap_options, index=colormap_options.index("viridis")),
    "Electromigration": st.sidebar.selectbox("Electromigration Free Energy Colormap", colormap_options, index=colormap_options.index("rdbu")),
    "Elastic": st.sidebar.selectbox("Elastic Free Energy Colormap", colormap_options, index=colormap_options.index("plasma")),
    "Total": st.sidebar.selectbox("Total Free Energy Colormap", colormap_options, index=colormap_options.index("jet")),
}

# Material properties
Vm_Sn = 16.29e-6
Vm_Li2Sn5 = 20.135e-6
NA = 6.022e23
e = 1.602e-19
Z_Li2Sn5 = 30
Z_Sn = 2
rho_Li2Sn5 = 1.75e-7
rho_Sn = 1.10e-7
W_mol = 1.629
W = W_mol / Vm_Sn  # 1e5 J/m³
G_Sn_mol = -0.402363
G_Li2Sn5_mol = -0.53757
E_Sn_mol = -0.4887
E_Li2Sn5_mol = -0.6501
G_Sn = G_Sn_mol / Vm_Sn  # -2.47e4 J/m³
G_Li2Sn5 = G_Li2Sn5_mol / Vm_Sn  # -3.3e4 J/m³
E_Sn = E_Sn_mol / Vm_Sn  # -3e4 J/m³
E_Li2Sn5 = E_Li2Sn5_mol / Vm_Sn  # -3.99e4 J/m³
c_Sn = 0.95
c_Li2Sn5 = 5/7
F = 96485
R = 8.314
T = 523.15
phi_0 = R * T / F
beta_2D = 0.5 * np.log(Vm_Li2Sn5 / Vm_Sn) / (1 - c_Li2Sn5)
epsilon = 1e-6

# Non-dimensional parameters
tau = Lx**2 / (M_Sn_dim * W)
t_max = t_max_dim / tau
beta = beta_dim / (W * Lx**2)
M_Li2Sn5 = M_Li2Sn5_dim / M_Sn_dim
M_Sn = 1.0
rho_Li2Sn5_star = rho_Li2Sn5 * F / (Lx * phi_0)
rho_Sn_star = rho_Sn * F / (Lx * phi_0)
t_start = min(1e-28 / tau, t_max * 1e-2)

# Grid
logger.info("Setting up grid...")
dx, dy = 1.0 / (Nx - 1), 1.0 / (Ny - 1)
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y)
N = Nx * Ny

# Initial conditions
logger.info("Initializing fields...")
c = np.ones((Ny, Nx)) * c_Sn
eta1 = np.ones((Ny, Nx))  # Sn matrix
eta2 = np.zeros((Ny, Nx))  # Top right Li₂Sn₅
eta3 = np.zeros((Ny, Nx))  # Bottom right Li₂Sn₅
seed_radii = [0.1, 0.1]
seed_centers = [(0.9, 0.7), (0.9, 0.3)]  # Top right, bottom right
for i, (radius, (center_x, center_y)) in enumerate(zip(seed_radii, seed_centers)):
    mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
    c[mask] = c_Li2Sn5
    eta1[mask] = 0.0
    if i == 0:
        eta2[mask] = 1.0
        eta3[mask] = 0.0
    else:
        eta2[mask] = 0.0
        eta3[mask] = 1.0

# Initial condition visualization
if st.checkbox("Show Initial Conditions"):
    logger.info("Displaying initial conditions...")
    for field, title, label in [
        (c, "Initial Sn Mole Fraction", "Sn Mole Fraction"),
        (eta1, "Initial Eta_1 (Sn Matrix)", "η₁"),
        (eta2, "Initial Eta_2 (Li₂Sn₅ Top Right)", "η₂"),
        (eta3, "Initial Eta_3 (Li₂Sn₅ Bottom Right)", "η₃")
    ]:
        fig = go.Figure(data=go.Heatmap(z=field, x=x*Lx*1e6, y=y*Ly*1e6, colorscale=colormaps["Concentration"]))
        fig.update_layout(title=title, xaxis_title="X (μm)", yaxis_title="Y (μm)", coloraxis_colorbar_title=label)
        st.plotly_chart(fig)

# Moelans interpolation function
def h(eta_i, eta1, eta2, eta3):
    eta_sum = eta1**2 + eta2**2 + eta3**2 + epsilon
    return eta_i**2 / eta_sum

# Free energy functions (dimensional, J/m³)
def f_chemical(c, eta1, eta2, eta3):
    h1 = h(eta1, eta1, eta2, eta3)
    h2 = h(eta2, eta1, eta2, eta3)
    h3 = h(eta3, eta1, eta2, eta3)
    g = 6 * (eta1**2 * (1 - eta1)**2 + eta2**2 * (1 - eta2)**2 + eta3**2 * (1 - eta3)**2)
    return (h1 * G_Sn_mol + (h2 + h3) * G_Li2Sn5_mol) * c / Vm_Sn + W_mol * (c - c_Sn) * (c - c_Li2Sn5) / Vm_Sn + W_mol * g / Vm_Sn

def f_electromigration(c, eta1, eta2, eta3, phi):
    h1 = h(eta1, eta1, eta2, eta3)
    h2 = h(eta2, eta1, eta2, eta3)
    h3 = h(eta3, eta1, eta2, eta3)
    return -gamma * (NA * e / Vm_Sn) * ((h2 + h3) * Z_Li2Sn5 + h1 * Z_Sn) * (1 - c) * (phi / phi_0)

def f_elastic(c, eta1, eta2, eta3):
    h1 = h(eta1, eta1, eta2, eta3)
    h2 = h(eta2, eta1, eta2, eta3)
    h3 = h(eta3, eta1, eta2, eta3)
    C_11 = (h2 + h3) * C_11_Li2Sn5 + h1 * C_11_Sn
    C_12 = (h2 + h3) * C_12_Li2Sn5 + h1 * C_12_Sn
    C_21 = (h2 + h3) * C_21_Li2Sn5 + h1 * C_21_Sn
    C_22 = (h2 + h3) * C_22_Li2Sn5 + h1 * C_22_Sn
    epsilon_xx = alpha_eigenstrain * beta_2D * (1 - c) * (h2 + h3)
    return ((0.5 * (C_11 + C_12 + C_21 + C_22) * epsilon_xx**2) / Vm_Sn + (h1 * E_Sn + (h2 + h3) * E_Li2Sn5) * c)

# Non-dimensional free energy functions
def f_chemical_nd(c, eta1, eta2, eta3):
    return f_chemical(c, eta1, eta2, eta3) / W

def f_electromigration_nd(c, eta1, eta2, eta3, phi):
    return f_electromigration(c, eta1, eta2, eta3, phi) / W

def f_elastic_nd(c, eta1, eta2, eta3):
    return f_elastic(c, eta1, eta2, eta3) / W

# Derivatives (non-dimensional)
def df_chemical_dc(c, eta1, eta2, eta3):
    h1 = h(eta1, eta1, eta2, eta3)
    h2 = h(eta2, eta1, eta2, eta3)
    h3 = h(eta3, eta1, eta2, eta3)
    return ((h1 * G_Sn_mol + (h2 + h3) * G_Li2Sn5_mol) + W_mol * (2 * c - c_Sn - c_Li2Sn5)) / (W * Vm_Sn)

def df_chemical_deta1(c, eta1, eta2, eta3):
    eta_sum = eta1**2 + eta2**2 + eta3**2 + epsilon
    dh1_deta1 = 2 * eta1 / eta_sum - 2 * eta1**3 / eta_sum**2
    dh2_deta1 = -2 * eta2**2 * eta1 / eta_sum**2
    dh3_deta1 = -2 * eta3**2 * eta1 / eta_sum**2
    g = 6 * (eta1**2 * (1 - eta1)**2 + eta2**2 * (1 - eta2)**2 + eta3**2 * (1 - eta3)**2)
    dg_deta1 = 6 * 2 * eta1 * (1 - eta1) * (1 - 2 * eta1)
    return ((G_Sn_mol * c * dh1_deta1 + G_Li2Sn5_mol * c * (dh2_deta1 + dh3_deta1)) + W_mol * dg_deta1) / (W * Vm_Sn)

def df_chemical_deta2(c, eta1, eta2, eta3):
    eta_sum = eta1**2 + eta2**2 + eta3**2 + epsilon
    dh1_deta2 = -2 * eta1**2 * eta2 / eta_sum**2
    dh2_deta2 = 2 * eta2 / eta_sum - 2 * eta2**3 / eta_sum**2
    dh3_deta2 = -2 * eta3**2 * eta2 / eta_sum**2
    g = 6 * (eta1**2 * (1 - eta1)**2 + eta2**2 * (1 - eta2)**2 + eta3**2 * (1 - eta3)**2)
    dg_deta2 = 6 * 2 * eta2 * (1 - eta2) * (1 - 2 * eta2)
    return ((G_Sn_mol * c * dh1_deta2 + G_Li2Sn5_mol * c * (dh2_deta2 + dh3_deta2)) + W_mol * dg_deta2) / (W * Vm_Sn)

def df_chemical_deta3(c, eta1, eta2, eta3):
    eta_sum = eta1**2 + eta2**2 + eta3**2 + epsilon
    dh1_deta3 = -2 * eta1**2 * eta3 / eta_sum**2
    dh2_deta3 = -2 * eta2**2 * eta3 / eta_sum**2
    dh3_deta3 = 2 * eta3 / eta_sum - 2 * eta3**3 / eta_sum**2
    g = 6 * (eta1**2 * (1 - eta1)**2 + eta2**2 * (1 - eta2)**2 + eta3**2 * (1 - eta3)**2)
    dg_deta3 = 6 * 2 * eta3 * (1 - eta3) * (1 - 2 * eta3)
    return ((G_Sn_mol * c * dh1_deta3 + G_Li2Sn5_mol * c * (dh2_deta3 + dh3_deta3)) + W_mol * dg_deta3) / (W * Vm_Sn)

def df_electromigration_dc(c, eta1, eta2, eta3, phi):
    h1 = h(eta1, eta1, eta2, eta3)
    h2 = h(eta2, eta1, eta2, eta3)
    h3 = h(eta3, eta1, eta2, eta3)
    return -gamma * (NA * e / (W * Vm_Sn)) * ((h2 + h3) * Z_Li2Sn5 + h1 * Z_Sn) * (-phi / phi_0)

def df_electromigration_deta1(c, eta1, eta2, eta3, phi):
    eta_sum = eta1**2 + eta2**2 + eta3**2 + epsilon
    dh1_deta1 = 2 * eta1 / eta_sum - 2 * eta1**3 / eta_sum**2
    dh2_deta1 = -2 * eta2**2 * eta1 / eta_sum**2
    dh3_deta1 = -2 * eta3**2 * eta1 / eta_sum**2
    return -gamma * (NA * e / (W * Vm_Sn)) * (1 - c) * (phi / phi_0) * (dh1_deta1 * Z_Sn + (dh2_deta1 + dh3_deta1) * Z_Li2Sn5)

def df_electromigration_deta2(c, eta1, eta2, eta3, phi):
    eta_sum = eta1**2 + eta2**2 + eta3**2 + epsilon
    dh1_deta2 = -2 * eta1**2 * eta2 / eta_sum**2
    dh2_deta2 = 2 * eta2 / eta_sum - 2 * eta2**3 / eta_sum**2
    dh3_deta2 = -2 * eta3**2 * eta2 / eta_sum**2
    return -gamma * (NA * e / (W * Vm_Sn)) * (1 - c) * (phi / phi_0) * (dh1_deta2 * Z_Sn + (dh2_deta2 + dh3_deta2) * Z_Li2Sn5)

def df_electromigration_deta3(c, eta1, eta2, eta3, phi):
    eta_sum = eta1**2 + eta2**2 + eta3**2 + epsilon
    dh1_deta3 = -2 * eta1**2 * eta3 / eta_sum**2
    dh2_deta3 = -2 * eta2**2 * eta3 / eta_sum**2
    dh3_deta3 = 2 * eta3 / eta_sum - 2 * eta3**3 / eta_sum**2
    return -gamma * (NA * e / (W * Vm_Sn)) * (1 - c) * (phi / phi_0) * (dh1_deta3 * Z_Sn + (dh2_deta3 + dh3_deta3) * Z_Li2Sn5)

def df_elastic_dc(c, eta1, eta2, eta3):
    h1 = h(eta1, eta1, eta2, eta3)
    h2 = h(eta2, eta1, eta2, eta3)
    h3 = h(eta3, eta1, eta2, eta3)
    C_11 = (h2 + h3) * C_11_Li2Sn5 + h1 * C_11_Sn
    C_12 = (h2 + h3) * C_12_Li2Sn5 + h1 * C_12_Sn
    C_21 = (h2 + h3) * C_21_Li2Sn5 + h1 * C_21_Sn
    C_22 = (h2 + h3) * C_22_Li2Sn5 + h1 * C_22_Sn
    epsilon_xx = alpha_eigenstrain * beta_2D * (1 - c) * (h2 + h3)
    return ((0.5 * (C_11 + C_12 + C_21 + C_22) * 2 * epsilon_xx * (-alpha_eigenstrain * beta_2D * (h2 + h3))) / Vm_Sn + (h1 * E_Sn + (h2 + h3) * E_Li2Sn5)) / W

def df_elastic_deta1(c, eta1, eta2, eta3):
    eta_sum = eta1**2 + eta2**2 + eta3**2 + epsilon
    dh1_deta1 = 2 * eta1 / eta_sum - 2 * eta1**3 / eta_sum**2
    dh2_deta1 = -2 * eta2**2 * eta1 / eta_sum**2
    dh3_deta1 = -2 * eta3**2 * eta1 / eta_sum**2
    C_11 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_11_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_11_Sn
    C_12 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_12_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_12_Sn
    C_21 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_21_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_21_Sn
    C_22 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_22_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_22_Sn
    dC_11_deta1 = (dh2_deta1 + dh3_deta1) * C_11_Li2Sn5 + dh1_deta1 * C_11_Sn
    dC_12_deta1 = (dh2_deta1 + dh3_deta1) * C_12_Li2Sn5 + dh1_deta1 * C_12_Sn
    dC_21_deta1 = (dh2_deta1 + dh3_deta1) * C_21_Li2Sn5 + dh1_deta1 * C_21_Sn
    dC_22_deta1 = (dh2_deta1 + dh3_deta1) * C_22_Li2Sn5 + dh1_deta1 * C_22_Sn
    epsilon_xx = alpha_eigenstrain * beta_2D * (1 - c) * (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3))
    depsilon_xx_deta1 = alpha_eigenstrain * beta_2D * (1 - c) * (dh2_deta1 + dh3_deta1)
    return ((0.5 * (dC_11_deta1 + dC_12_deta1 + dC_21_deta1 + dC_22_deta1) * epsilon_xx**2 + 
             (C_11 + C_12 + C_21 + C_22) * epsilon_xx * depsilon_xx_deta1) / Vm_Sn + 
            (dh1_deta1 * E_Sn + (dh2_deta1 + dh3_deta1) * E_Li2Sn5) * c) / W

def df_elastic_deta2(c, eta1, eta2, eta3):
    eta_sum = eta1**2 + eta2**2 + eta3**2 + epsilon
    dh1_deta2 = -2 * eta1**2 * eta2 / eta_sum**2
    dh2_deta2 = 2 * eta2 / eta_sum - 2 * eta2**3 / eta_sum**2
    dh3_deta2 = -2 * eta3**2 * eta2 / eta_sum**2
    C_11 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_11_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_11_Sn
    C_12 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_12_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_12_Sn
    C_21 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_21_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_21_Sn
    C_22 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_22_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_22_Sn
    dC_11_deta2 = (dh2_deta2 + dh3_deta2) * C_11_Li2Sn5 + dh1_deta2 * C_11_Sn
    dC_12_deta2 = (dh2_deta2 + dh3_deta2) * C_12_Li2Sn5 + dh1_deta2 * C_12_Sn
    dC_21_deta2 = (dh2_deta2 + dh3_deta2) * C_21_Li2Sn5 + dh1_deta2 * C_21_Sn
    dC_22_deta2 = (dh2_deta2 + dh3_deta2) * C_22_Li2Sn5 + dh1_deta2 * C_22_Sn
    epsilon_xx = alpha_eigenstrain * beta_2D * (1 - c) * (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3))
    depsilon_xx_deta2 = alpha_eigenstrain * beta_2D * (1 - c) * (dh2_deta2 + dh3_deta2)
    return ((0.5 * (dC_11_deta2 + dC_12_deta2 + dC_21_deta2 + dC_22_deta2) * epsilon_xx**2 + 
             (C_11 + C_12 + C_21 + C_22) * epsilon_xx * depsilon_xx_deta2) / Vm_Sn + 
            (dh1_deta2 * E_Sn + (dh2_deta2 + dh3_deta2) * E_Li2Sn5) * c) / W

def df_elastic_deta3(c, eta1, eta2, eta3):
    eta_sum = eta1**2 + eta2**2 + eta3**2 + epsilon
    dh1_deta3 = -2 * eta1**2 * eta3 / eta_sum**2
    dh2_deta3 = -2 * eta2**2 * eta3 / eta_sum**2
    dh3_deta3 = 2 * eta3 / eta_sum - 2 * eta3**3 / eta_sum**2
    C_11 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_11_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_11_Sn
    C_12 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_12_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_12_Sn
    C_21 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_21_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_21_Sn
    C_22 = (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3)) * C_22_Li2Sn5 + h(eta1, eta1, eta2, eta3) * C_22_Sn
    dC_11_deta3 = (dh2_deta3 + dh3_deta3) * C_11_Li2Sn5 + dh1_deta3 * C_11_Sn
    dC_12_deta3 = (dh2_deta3 + dh3_deta3) * C_12_Li2Sn5 + dh1_deta3 * C_12_Sn
    dC_21_deta3 = (dh2_deta3 + dh3_deta3) * C_21_Li2Sn5 + dh1_deta3 * C_21_Sn
    dC_22_deta3 = (dh2_deta3 + dh3_deta3) * C_22_Li2Sn5 + dh1_deta3 * C_22_Sn
    epsilon_xx = alpha_eigenstrain * beta_2D * (1 - c) * (h(eta2, eta1, eta2, eta3) + h(eta3, eta1, eta2, eta3))
    depsilon_xx_deta3 = alpha_eigenstrain * beta_2D * (1 - c) * (dh2_deta3 + dh3_deta3)
    return ((0.5 * (dC_11_deta3 + dC_12_deta3 + dC_21_deta3 + dC_22_deta3) * epsilon_xx**2 + 
             (C_11 + C_12 + C_21 + C_22) * epsilon_xx * depsilon_xx_deta3) / Vm_Sn + 
            (dh1_deta3 * E_Sn + (dh2_deta3 + dh3_deta3) * E_Li2Sn5) * c) / W

# Numerical solver
def laplacian(field, dx, dy):
    lap = np.zeros_like(field)
    lap[1:-1, 1:-1] = (field[1:-1, 2:] + field[1:-1, :-2] - 2 * field[1:-1, 1:-1]) / dx**2 + \
                      (field[2:, 1:-1] + field[:-2, 1:-1] - 2 * field[1:-1, 1:-1]) / dy**2
    return lap

def gradient(field, dx, dy):
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    grad_x[1:-1, 1:-1] = (field[1:-1, 2:] - field[1:-1, :-2]) / (2 * dx)
    grad_y[1:-1, 1:-1] = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2 * dy)
    return grad_x, grad_y

def solve_potential(eta1, eta2, eta3):
    h1 = h(eta1, eta1, eta2, eta3)
    h2 = h(eta2, eta1, eta2, eta3)
    h3 = h(eta3, eta1, eta2, eta3)
    conductivity = 1 / (np.maximum((h2 + h3) * rho_Li2Sn5_star + h1 * rho_Sn_star, 1e-10))
    n_interior = (Ny-2) * (Nx-2)
    A = diags([
        -conductivity[1:-1, 1:-1].flatten() / dx**2,  # Left neighbor
        -conductivity[1:-1, 1:-1].flatten() / dy**2,  # Bottom neighbor
        2 * conductivity[1:-1, 1:-1].flatten() * (1/dx**2 + 1/dy**2),  # Center
        -conductivity[1:-1, 1:-1].flatten() / dx**2,  # Right neighbor
        -conductivity[1:-1, 1:-1].flatten() / dy**2   # Top neighbor
    ], offsets=[-Nx, -1, 0, 1, Nx], shape=(n_interior, n_interior))
    phi = np.zeros((Ny, Nx))
    phi[0, :] = 0  # Dirichlet BC at y=0
    phi[-1, :] = phi_anode / phi_0  # Dirichlet BC at y=Ly
    phi_flat = phi[1:-1, 1:-1].flatten()
    phi[1:-1, 1:-1] = spsolve(A, np.zeros_like(phi_flat)).reshape(Ny-2, Nx-2)
    return phi

def system(t, u):
    c = u[:N].reshape(Ny, Nx)
    eta1 = u[N:2*N].reshape(Ny, Nx)
    eta2 = u[2*N:3*N].reshape(Ny, Nx)
    eta3 = u[3*N:4*N].reshape(Ny, Nx)
    phi = solve_potential(eta1, eta2, eta3)
    
    h1 = h(eta1, eta1, eta2, eta3)
    h2 = h(eta2, eta1, eta2, eta3)
    h3 = h(eta3, eta1, eta2, eta3)
    M = (h2 + h3) * M_Li2Sn5 + h1 * M_Sn
    
    mu = df_chemical_dc(c, eta1, eta2, eta3) + df_electromigration_dc(c, eta1, eta2, eta3, phi) + df_elastic_dc(c, eta1, eta2, eta3)
    grad_mu_x, grad_mu_y = gradient(mu, dx, dy)
    dc_dt = np.zeros_like(c)
    dc_dt[1:-1, 1:-1] = (M[1:-1, 2:] * grad_mu_x[1:-1, 2:] - M[1:-1, :-2] * grad_mu_x[1:-1, :-2]) / (2 * dx) + \
                        (M[2:, 1:-1] * grad_mu_y[2:, 1:-1] - M[:-2, 1:-1] * grad_mu_y[:-2, 1:-1]) / (2 * dy)
    
    deta1_dt = -L * (df_chemical_deta1(c, eta1, eta2, eta3) + df_electromigration_deta1(c, eta1, eta2, eta3, phi) + 
                    df_elastic_deta1(c, eta1, eta2, eta3) - beta * laplacian(eta1, dx, dy))
    deta2_dt = -L * (df_chemical_deta2(c, eta1, eta2, eta3) + df_electromigration_deta2(c, eta1, eta2, eta3, phi) + 
                    df_elastic_deta2(c, eta1, eta2, eta3) - beta * laplacian(eta2, dx, dy))
    deta3_dt = -L * (df_chemical_deta3(c, eta1, eta2, eta3) + df_electromigration_deta3(c, eta1, eta2, eta3, phi) + 
                    df_elastic_deta3(c, eta1, eta2, eta3) - beta * laplacian(eta3, dx, dy))
    
    return np.concatenate([dc_dt.flatten(), deta1_dt.flatten(), deta2_dt.flatten(), deta3_dt.flatten()])

# Free energy visualization
st.header("Free Energy Visualization")
c_grid = np.linspace(0, 1, 50)
eta1_grid = np.linspace(0, 1, 50)
C, ETA1 = np.meshgrid(c_grid, eta1_grid)
ETA2 = 1 - ETA1
ETA3 = np.zeros_like(ETA1)
PHI = phi_anode / phi_0
F_CHEM = f_chemical(C, ETA1, ETA2, ETA3)
F_ELEC = f_electromigration(C, ETA1, ETA2, ETA3, PHI)
F_ELASTIC = f_elastic(C, ETA1, ETA2, ETA3)
F_TOTAL = F_CHEM + F_ELEC + F_ELASTIC

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}],
           [{"type": "surface"}, {"type": "contour"}]],
    subplot_titles=("Chemical Free Energy", "Electromigration Free Energy", "Elastic Free Energy", "Total Free Energy (Contour)")
)

fig.add_trace(
    go.Surface(x=c_grid, y=eta1_grid, z=F_CHEM, colorscale=colormaps["Chemical"], showscale=True),
    row=1, col=1
)
fig.add_trace(
    go.Scatter3d(x=[c_Sn, c_Li2Sn5], y=[1, 0], z=[f_chemical(c_Sn, 1, 0, 0), f_chemical(c_Li2Sn5, 0, 1, 0)], mode="markers", marker=dict(size=5, color="red")),
    row=1, col=1
)

fig.add_trace(
    go.Surface(x=c_grid, y=eta1_grid, z=F_ELEC, colorscale=colormaps["Electromigration"], showscale=True),
    row=1, col=2
)
fig.add_trace(
    go.Scatter3d(x=[c_Sn, c_Li2Sn5], y=[1, 0], z=[f_electromigration(c_Sn, 1, 0, 0, PHI), f_electromigration(c_Li2Sn5, 0, 1, 0, PHI)], mode="markers", marker=dict(size=5, color="red")),
    row=1, col=2
)

fig.add_trace(
    go.Surface(x=c_grid, y=eta1_grid, z=F_ELASTIC, colorscale=colormaps["Elastic"], showscale=True),
    row=2, col=1
)
fig.add_trace(
    go.Scatter3d(x=[c_Sn, c_Li2Sn5], y=[1, 0], z=[f_elastic(c_Sn, 1, 0, 0), f_elastic(c_Li2Sn5, 0, 1, 0)], mode="markers", marker=dict(size=5, color="red")),
    row=2, col=1
)

fig.add_trace(
    go.Contour(x=c_grid, y=eta1_grid, z=F_TOTAL, colorscale=colormaps["Total"], showscale=True),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=[c_Sn, c_Li2Sn5], y=[1, 0], mode="markers", marker=dict(size=10, color="red", symbol="circle")),
    row=2, col=2
)

fig.update_layout(
    height=800,
    showlegend=False,
    scene=dict(xaxis_title="c (Sn Mole Fraction)", yaxis_title="η₁ (Sn Phase)", zaxis_title="Energy (J/m³)"),
    scene2=dict(xaxis_title="c", yaxis_title="η₁", zaxis_title="Energy (J/m³)"),
    scene3=dict(xaxis_title="c", yaxis_title="η₁", zaxis_title="Energy (J/m³)")
)
fig.update_scenes(aspectmode="cube")
st.plotly_chart(fig, use_container_width=True)

# Solve
logger.info("Solving system...")
u0 = np.concatenate([c.flatten(), eta1.flatten(), eta2.flatten(), eta3.flatten()])
t_span = (t_start, t_max)
t_eval = np.linspace(t_start, t_max, num_t_eval)
sol = solve_ivp(system, t_span, u0, method='RK45', t_eval=t_eval)

# Post-processing
c_t = sol.y[:N, :].reshape(num_t_eval, Ny, Nx)
eta1_t = sol.y[N:2*N, :].reshape(num_t_eval, Ny, Nx)
eta2_t = sol.y[2*N:3*N, :].reshape(num_t_eval, Ny, Nx)
eta3_t = sol.y[3*N:4*N, :].reshape(num_t_eval, Ny, Nx)
phi_t = np.array([solve_potential(eta1_t[i], eta2_t[i], eta3_t[i]) for i in range(num_t_eval)])

# Calculate strains and stresses
strain_xx_t = np.zeros((num_t_eval, Ny, Nx))
stress_xx_t = np.zeros((num_t_eval, Ny, Nx))
stress_yy_t = np.zeros((num_t_eval, Ny, Nx))
von_mises_t = np.zeros((num_t_eval, Ny, Nx))
for i in range(num_t_eval):
    h1 = h(eta1_t[i], eta1_t[i], eta2_t[i], eta3_t[i])
    h2 = h(eta2_t[i], eta1_t[i], eta2_t[i], eta3_t[i])
    h3 = h(eta3_t[i], eta1_t[i], eta2_t[i], eta3_t[i])
    C_11 = (h2 + h3) * C_11_Li2Sn5 + h1 * C_11_Sn
    C_12 = (h2 + h3) * C_12_Li2Sn5 + h1 * C_12_Sn
    C_21 = (h2 + h3) * C_21_Li2Sn5 + h1 * C_21_Sn
    C_22 = (h2 + h3) * C_22_Li2Sn5 + h1 * C_22_Sn
    strain_xx_t[i] = alpha_eigenstrain * beta_2D * (1 - c_t[i]) * (h2 + h3)
    stress_xx_t[i] = C_11 * strain_xx_t[i] + C_12 * strain_xx_t[i]
    stress_yy_t[i] = C_21 * strain_xx_t[i] + C_22 * strain_xx_t[i]
    von_mises_t[i] = np.sqrt(0.5 * ((stress_xx_t[i] - stress_yy_t[i])**2 + stress_xx_t[i]**2 + stress_yy_t[i]**2))

# Visualization
st.header("Simulation Results")
time_idx = st.slider("Select time step", 0, num_t_eval-1, num_t_eval-1)
fields = {
    "Concentration": (c_t[time_idx], "Sn Mole Fraction"),
    "Eta_1": (eta1_t[time_idx], "η₁ (Sn Matrix)"),
    "Eta_2": (eta2_t[time_idx], "η₂ (Li₂Sn₅ Top Right)"),
    "Eta_3": (eta3_t[time_idx], "η₃ (Li₂Sn₅ Bottom Right)"),
    "Potential": (phi_t[time_idx], "Potential (V)"),
    "Strain_xx": (strain_xx_t[time_idx], "Strain xx"),
    "Stress_xx": (stress_xx_t[time_idx], "Stress xx (Pa)"),
    "Stress_yy": (stress_yy_t[time_idx], "Stress yy (Pa)"),
    "Von_Mises_Stress": (von_mises_t[time_idx], "Von Mises Stress (Pa)")
}

for name, (field, label) in fields.items():
    fig = go.Figure(data=go.Heatmap(z=field, x=x*Lx*1e6, y=y*Ly*1e6, colorscale=colormaps[name]))
    fig.update_layout(title=f"{name} at t = {sol.t[time_idx]*tau:.2e} s", xaxis_title="X (μm)", yaxis_title="Y (μm)", coloraxis_colorbar_title=label)
    st.plotly_chart(fig)

# Equilibrium analysis
st.header("Equilibrium Analysis")
f_chem_sn = f_chemical(c_Sn, 1, 0, 0)
f_elec_sn = f_electromigration(c_Sn, 1, 0, 0, phi_anode / phi_0)
f_elastic_sn = f_elastic(c_Sn, 1, 0, 0)
f_total_sn = f_chem_sn + f_elec_sn + f_elastic_sn
f_chem_li2sn5 = f_chemical(c_Li2Sn5, 0, 1, 0)
f_elec_li2sn5 = f_electromigration(c_Li2Sn5, 0, 1, 0, phi_anode / phi_0)
f_elastic_li2sn5 = f_elastic(c_Li2Sn5, 0, 1, 0)
f_total_li2sn5 = f_chem_li2sn5 + f_elec_li2sn5 + f_elastic_li2sn5

st.write(f"At φ = {phi_anode:.3f} V:")
st.write(f"Sn Total Free Energy: {f_total_sn:.2e} J/m³")
st.write(f"Li₂Sn₅ Total Free Energy: {f_total_li2sn5:.2e} J/m³")
mu_sn = df_chemical_dc(c_Sn, 1, 0, 0) + df_electromigration_dc(c_Sn, 1, 0, 0, phi_anode / phi_0) + df_elastic_dc(c_Sn, 1, 0, 0)
mu_li2sn5 = df_chemical_dc(c_Li2Sn5, 0, 1, 0) + df_electromigration_dc(c_Li2Sn5, 0, 1, 0, phi_anode / phi_0) + df_elastic_dc(c_Li2Sn5, 0, 1, 0)
st.write(f"Chemical Potential at Sn: {mu_sn*W:.2e} J/m³")
st.write(f"Chemical Potential at Li₂Sn₅: {mu_li2sn5*W:.2e} J/m³")
if mu_li2sn5 < mu_sn:
    st.success("Li₂Sn₅ has lower chemical potential, favored during lithiation.")
else:
    st.warning("Li₂Sn₅ not favored. Adjust parameters.")

# Save results
if st.button("Save Results"):
    with h5py.File("simulation_results.h5", "w") as f:
        f.create_dataset("c", data=c_t)
        f.create_dataset("eta1", data=eta1_t)
        f.create_dataset("eta2", data=eta2_t)
        f.create_dataset("eta3", data=eta3_t)
        f.create_dataset("phi", data=phi_t)
        f.create_dataset("strain_xx", data=strain_xx_t)
        f.create_dataset("stress_xx", data=stress_xx_t)
        f.create_dataset("stress_yy", data=stress_yy_t)
        f.create_dataset("von_mises", data=von_mises_t)
        f.create_dataset("time", data=sol.t * tau)
    st.success("Results saved to simulation_results.h5")

# Display logs
if log_output:
    st.header("Simulation Logs")
    st.text("\n".join(log_output))
