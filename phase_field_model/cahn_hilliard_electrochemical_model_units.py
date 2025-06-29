import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pyvista as pv
import h5py
from vtk import vtkRectilinearGridWriter
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

# Streamlit interface
st.title("Cahn-Hilliard Simulation for Sn → Li₂Sn₅ Phase Transformation")
st.markdown("Simulate the phase transformation with electromigration and elastic effects.")

# Parameters (configurable via sliders and number inputs)
st.sidebar.header("Simulation Parameters")
Lx = st.sidebar.slider("Domain size X (m)", 1e-6, 3e-6, 1.5e-6, step=1e-7, format="%.1e")
Ly = st.sidebar.slider("Domain size Y (m)", 1e-6, 3e-6, 1.5e-6, step=1e-7, format="%.1e")
Nx = st.sidebar.slider("Grid points X", 20, 100, 20)  # Reduced for faster testing
Ny = st.sidebar.slider("Grid points Y", 20, 100, 20)
t_max = st.sidebar.number_input("Simulation time (s)", min_value=1e-28, max_value=1e-16, value=1e-25, step=1e-28, format="%.1e")
num_t_eval = st.sidebar.slider("Number of output time points", 3, 10, 5)
kappa = st.sidebar.number_input("Gradient energy coefficient (J/m)", min_value=1e-12, max_value=1e-8, value=1e-8, step=1e-12, format="%.1e")  # Increased
M_Li2Sn5 = st.sidebar.number_input("Mobility Li₂Sn₅ (m²/(J·s))", min_value=1e-12, max_value=1e-8, value=1e-12, step=1e-12, format="%.1e")  # Reduced
M_Sn = st.sidebar.number_input("Mobility Sn (m²/(J·s))", min_value=1e-12, max_value=1e-8, value=1e-12, step=1e-12, format="%.1e")  # Reduced
j0 = st.sidebar.number_input("Exchange current density (A/m²)", min_value=1e3, max_value=1e5, value=1e4, step=1e3, format="%.1e")

# Fixed material properties
Vm_Sn = 16.29e-6
Vm_Li2Sn5 = 20.135e-6
NA = 6.022e23
e = 1.602e-19
Z_Li2Sn5 = 30
Z_Sn = 2
rho_Li2Sn5 = 1.75e-7
rho_Sn = 1.10e-7
C = 50e9
W = 1e5
beta_sigmoid = 20
c0 = 5/7
alpha = 0.5
F = 96485
R = 8.314
T = 523.15
phi_eq = 0
beta_2D = (1/2) * np.log(Vm_Li2Sn5 / Vm_Sn) / (1 - 5/7)

# Grid
logger.info("Setting up grid...")
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition with larger Li2Sn5 seeds
logger.info("Initializing concentration field...")
c = np.ones((Ny, Nx)) * 0.95
seed_radii = [0.1 * Lx, 0.12 * Lx]
seed_centers = [(0.9 * Lx, 0.3 * Ly), (0.9 * Lx, 0.7 * Ly)]
for radius, (center_x, center_y) in zip(seed_radii, seed_centers):
    mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
    c[mask] = 5/7

# Initial condition visualization
if st.checkbox("Show Initial Condition"):
    logger.info("Displaying initial condition...")
    fig = go.Figure(data=go.Heatmap(z=c, x=x*1e6, y=y*1e6, colorscale="Viridis"))
    fig.update_layout(title="Initial Sn Mole Fraction", xaxis_title="X (μm)", yaxis_title="Y (μm)", coloraxis_colorbar_title="Sn Mole Fraction")
    st.plotly_chart(fig)

# Sigmoidal interpolation
def h(c):
    return 1 / (1 + np.exp(-beta_sigmoid * (c - c0)))

# Free energy functions
def f_chemical(c):
    G_Li2Sn5 = -2.95e4
    G_Sn = -2.47e4
    return (h(c) * G_Li2Sn5 + (1 - h(c)) * G_Sn) / Vm_Sn + W * c * (1 - c)

def f_electromigration(c, phi):
    return (NA * e / Vm_Sn) * (h(c) * Z_Li2Sn5 * (1 - c) + (1 - h(c)) * Z_Sn * (1 - c)) * phi

def f_elastic(c):
    epsilon_xx = beta_2D * (1 - c)
    return C * (epsilon_xx**2)

# Derivatives
def df_chemical_dc(c):
    dh_dc = beta_sigmoid * h(c) * (1 - h(c))
    G_Li2Sn5 = -2.95e4
    G_Sn = -2.47e4
    return (dh_dc * (G_Li2Sn5 - G_Sn) + W * (1 - 2 * c)) / Vm_Sn

def df_electromigration_dc(c, phi):
    dh_dc = beta_sigmoid * h(c) * (1 - h(c))
    return (NA * e / Vm_Sn) * (dh_dc * (Z_Li2Sn5 - Z_Sn) * (1 - c) - (h(c) * Z_Li2Sn5 + (1 - h(c)) * Z_Sn)) * phi

def df_elastic_dc(c):
    epsilon_xx = beta_2D * (1 - c)
    return -2 * C * epsilon_xx * beta_2D

# Mobility
def M(c):
    return h(c) * M_Li2Sn5 + (1 - h(c)) * M_Sn

# 2D Laplacian
def laplacian_2d(u, dx, dy):
    lap = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2
    lap += (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dy**2
    return lap

# CFL condition for adaptive time step
def compute_dt(c, dx, dy):
    M_max = np.max(M(c))
    dt = 0.5 * (dx**2 * dy**2) / (2 * M_max * kappa * (4 / dx**2 + 4 / dy**2))
    logger.info(f"Computed CFL time step: {dt:.1e} s")
    return dt

# Solve electric potential with sparse solver
def solve_potential(c, j_anode):
    logger.info("Solving electric potential...")
    kappa = 1 / (h(c) * rho_Li2Sn5 + (1 - h(c)) * rho_Sn)
    N = Nx * Ny
    A = np.zeros((N, N))
    b = np.zeros(N)
    for j in range(Ny):
        for i in range(Nx):
            idx = j * Nx + i
            if i == 0:  # Cathode (x = 0)
                A[idx, idx] = 1
                b[idx] = 0
            elif i == Nx - 1:  # Anode (x = Lx)
                A[idx, idx] = 1
                A[idx, idx - 1] = -1
                b[idx] = j_anode / kappa[j, i] * dx
            elif j == 0:  # Insulating (y = 0)
                A[idx, idx] = 1
                A[idx, idx + Nx] = -1 if j + 1 < Ny else 0
            elif j == Ny - 1:  # Insulating (y = Ly)
                A[idx, idx] = 1
                A[idx, idx - Nx] = -1 if j - 1 >= 0 else 0
            else:  # Interior
                A[idx, idx] = -2 * kappa[j, i] * (1 / dx**2 + 1 / dy**2)
                A[idx, idx - 1] = kappa[j, i] / dx**2 if i - 1 >= 0 else 0
                A[idx, idx + 1] = kappa[j, i] / dx**2 if i + 1 < Nx else 0
                A[idx, idx - Nx] = kappa[j, i] / dy**2 if j - 1 >= 0 else 0
                A[idx, idx + Nx] = kappa[j, i] / dy**2 if j + 1 < Ny else 0
    from scipy.sparse import csr_matrix
    A = csr_matrix(A)
    return spsolve(A, b).reshape((Ny, Nx))

# Butler-Volmer
def j_butler_volmer(phi):
    eta = phi - phi_eq
    return j0 * (np.exp(alpha * F * eta / (R * T)) - np.exp(-(1 - alpha) * F * eta / (R * T)))

# Cahn-Hilliard RHS
def cahn_hilliard(t, c_flat):
    c = c_flat.reshape((Ny, Nx))
    phi_anode = solve_potential(c, 0)[0, -1]
    phi = solve_potential(c, j_butler_volmer(phi_anode))
    mu = df_chemical_dc(c) + df_electromigration_dc(c, phi) + df_elastic_dc(c) - kappa * laplacian_2d(c, dx, dy)
    flux_x = M(c) * (np.roll(mu, -1, axis=1) - np.roll(mu, 1, axis=1)) / (2 * dx)
    flux_y = M(c) * (np.roll(mu, -1, axis=0) - np.roll(mu, 1, axis=0)) / (2 * dy)
    dc_dt = (np.roll(flux_x, -1, axis=1) - np.roll(flux_x, 1, axis=1)) / (2 * dx) + \
            (np.roll(flux_y, -1, axis=0) - np.roll(flux_y, 1, axis=0)) / (2 * dy)
    dc_dt[0, :] = dc_dt[-1, :] = 0
    dc_dt[:, 0] = dc_dt[:, -1] = 0
    return dc_dt.flatten()

# Postprocessing
def compute_variables(c, phi):
    epsilon_xx = beta_2D * (1 - c)
    sigma_xx = C * epsilon_xx
    return {"c": c, "phi": phi, "epsilon_xx": epsilon_xx, "sigma_xx": sigma_xx}

# Visualization functions
def plot_plotly(data, title, label):
    fig = go.Figure(data=go.Heatmap(z=data, x=x*1e6, y=y*1e6, colorscale="Viridis"))
    fig.update_layout(title=title, xaxis_title="X (μm)", yaxis_title="Y (μm)", coloraxis_colorbar_title=label)
    st.plotly_chart(fig)

def plot_matplotlib(data, title, label, filename):
    plt.figure()
    plt.imshow(data, extent=[0, Lx*1e6, 0, Ly*1e6], origin="lower", cmap="viridis")
    plt.colorbar(label=label)
    plt.title(title)
    plt.xlabel("X (μm)")
    plt.ylabel("Y (μm)")
    plt.savefig(filename)
    st.image(filename)

# Save to VTR
def save_vtr(c, phi, epsilon_xx, sigma_xx, t):
    logger.info(f"Saving VTR file for t = {t:.1e} s")
    grid = pv.RectilinearGrid(x, y)
    grid["Concentration"] = c.flatten(order="F")
    grid["Potential"] = phi.flatten(order="F")
    grid["Strain_xx"] = epsilon_xx.flatten(order="F")
    grid["Stress_xx"] = sigma_xx.flatten(order="F")
    grid.save(f"output_t_{t:.1e}.vtr")

# Save to HDF5
def save_h5(c, phi, epsilon_xx, sigma_xx, t):
    logger.info(f"Saving HDF5 file for t = {t:.1e} s")
    with h5py.File(f"output_t_{t:.1e}.h5", "w") as f:
        f.create_dataset("Concentration", data=c)
        f.create_dataset("Potential", data=phi)
        f.create_dataset("Strain_xx", data=epsilon_xx)
        f.create_dataset("Stress_xx", data=sigma_xx)

# PyVista visualization
def plot_pyvista(c, phi, epsilon_xx, sigma_xx):
    logger.info("Generating PyVista plot...")
    grid = pv.RectilinearGrid(x, y)
    grid["Concentration"] = c.flatten(order="F")
    grid["Potential"] = phi.flatten(order="F")
    grid["Strain_xx"] = epsilon_xx.flatten(order="F")
    grid["Stress_xx"] = sigma_xx.flatten(order="F")
    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(grid, scalars="Concentration", cmap="viridis")
    plotter.show(jupyter_backend="static", return_viewer=True)
    st.write("PyVista Concentration Plot (static)")
    st.image(plotter.image)

# Run simulation
if st.button("Run Simulation"):
    if t_max < 1e-28:
        st.error("Simulation time (t_max) must be at least 1e-28 s.")
    else:
        log_output.clear()
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        log_area = st.text_area("Simulation Log", value="", height=200)
        
        def progress_callback(t, c_flat):
            progress = min((t - 1e-28) / (t_max - 1e-28), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Simulation progress: {progress*100:.1f}%")
            log_area.value = "\n".join(log_output[-10:])
        
        with st.spinner("Running simulation..."):
            logger.info("Starting simulation...")
            c_flat = c.flatten()
            t_eval = np.logspace(np.log10(1e-28), np.log10(t_max), num_t_eval)
            dt_initial = compute_dt(c, dx, dy)
            st.write(f"Initial CFL Time Step: {dt_initial:.1e} s")
            sol = solve_ivp(
                cahn_hilliard, (1e-28, t_max), c_flat, t_eval=t_eval, method="BDF",
                atol=1e-6, rtol=1e-6, max_step=dt_initial
            )
            
            logger.info("Postprocessing results...")
            for i, t in enumerate(t_eval):
                c = sol.y[:, i].reshape((Ny, Nx))
                phi = solve_potential(c, j_butler_volmer(solve_potential(c, 0)[0, -1]))
                vars = compute_variables(c, phi)
                
                st.subheader(f"Results at t = {t:.1e} s")
                plot_plotly(vars["c"], f"Concentration at t = {t:.1e} s", "Sn Mole Fraction")
                plot_matplotlib(vars["c"], f"Concentration at t = {t:.1e} s", "Sn Mole Fraction", f"conc_t_{t:.1e}.png")
                plot_plotly(vars["phi"], f"Potential at t = {t:.1e} s", "Potential (V)")
                plot_matplotlib(vars["phi"], f"Potential at t = {t:.1e} s", "Potential (V)", f"phi_t_{t:.1e}.png")
                plot_plotly(vars["epsilon_xx"], f"Strain at t = {t:.1e} s", "Strain_xx")
                plot_matplotlib(vars["epsilon_xx"], f"Strain at t = {t:.1e} s", "Strain_xx", f"strain_t_{t:.1e}.png")
                plot_plotly(vars["sigma_xx"], f"Stress at t = {t:.1e} s", "Stress_xx (Pa)")
                plot_matplotlib(vars["sigma_xx"], f"Stress at t = {t:.1e} s", "Stress_xx (Pa)", f"stress_t_{t:.1e}.png")
                
                save_vtr(vars["c"], vars["phi"], vars["epsilon_xx"], vars["sigma_xx"], t)
                save_h5(vars["c"], vars["phi"], vars["epsilon_xx"], vars["sigma_xx"], t)
                plot_pyvista(vars["c"], vars["phi"], vars["epsilon_xx"], vars["sigma_xx"])
            
            progress_bar.progress(1.0)
            status_text.text("Simulation complete!")
            log_area.value = "\n".join(log_output[-10:])
            st.success("Simulation complete! Outputs saved as .vtr and .h5 files.")