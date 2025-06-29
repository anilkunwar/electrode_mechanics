import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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

# Configure PyVista for trame backend
pv.global_theme.show_scalar_bar = True
pv.global_theme.jupyter_backend = "trame"

# Streamlit interface
st.title("Non-Dimensional Cahn-Hilliard Simulation for Sn → Li₂Sn₅")
st.markdown("Simulate phase transformation with electromigration and elastic effects (non-dimensional).")

# Parameters
st.sidebar.header("Simulation Parameters")
Lx = st.sidebar.slider("Domain size X (m)", 1e-6, 3e-6, 1.5e-6, step=1e-7, format="%.1e")
Ly = st.sidebar.slider("Domain size Y (m)", 1e-6, 3e-6, 1.5e-6, step=1e-7, format="%.1e")
Nx = st.sidebar.slider("Grid points X", 20, 100, 20)
Ny = st.sidebar.slider("Grid points Y", 20, 100, 20)
t_max_dim = st.sidebar.number_input("Simulation time (s)", min_value=1e-28, max_value=1e-16, value=1e-17, step=1e-28, format="%.1e")
num_t_eval = st.sidebar.slider("Number of output time points", 3, 10, 5)
kappa_dim = st.sidebar.number_input("Gradient energy coefficient (J/m)", min_value=1e-12, max_value=1e-8, value=1e-8, step=1e-12, format="%.1e")
M_Li2Sn5_dim = st.sidebar.number_input("Mobility Li₂Sn₅ (m²/(J·s))", min_value=1e-12, max_value=1e-8, value=1e-12, step=1e-12, format="%.1e")
M_Sn_dim = st.sidebar.number_input("Mobility Sn (m²/(J·s))", min_value=1e-12, max_value=1e-8, value=1e-12, step=1e-12, format="%.1e")

# Fixed material properties
Vm_Sn = 16.29e-6
Vm_Li2Sn5 = 20.135e-6
NA = 6.022e23
e = 1.602e-19
Z_Li2Sn5 = 30
Z_Sn = 2
rho_Li2Sn5 = 1.75e-7
rho_Sn = 1.10e-7
C_dim = 1e9  # 1 GPa
W = 1e5
beta_sigmoid = 20
c0 = 5/7
F = 96485
R = 8.314
T = 523.15
phi_0 = R * T / F
beta_2D = (1/2) * np.log(Vm_Li2Sn5 / Vm_Sn) / (1 - 5/7)

# Non-dimensional parameters
tau = Lx**2 / (M_Sn_dim * W)
t_max = t_max_dim / tau
kappa = kappa_dim / (W * Lx**2)
M_Li2Sn5 = M_Li2Sn5_dim / M_Sn_dim
M_Sn = 1.0
C = C_dim / W
rho_Li2Sn5_star = rho_Li2Sn5 * F / (Lx * phi_0)
rho_Sn_star = rho_Sn * F / (Lx * phi_0)
t_start = min(1e-28 / tau, t_max * 1e-2)  # Ensure t_start < t_max

# Grid
logger.info("Setting up grid...")
dx, dy = 1.0 / (Nx - 1), 1.0 / (Ny - 1)
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y)
N = Nx * Ny

# Initial condition
logger.info("Initializing concentration field...")
c = np.ones((Ny, Nx)) * 0.95
seed_radii = [0.1, 0.12]
seed_centers = [(0.9, 0.3), (0.9, 0.7)]
for radius, (center_x, center_y) in zip(seed_radii, seed_centers):
    mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
    c[mask] = 5/7

# Initial condition visualization
if st.checkbox("Show Initial Condition"):
    logger.info("Displaying initial condition...")
    fig = go.Figure(data=go.Heatmap(z=c, x=x*Lx*1e6, y=y*Ly*1e6, colorscale="Viridis"))
    fig.update_layout(title="Initial Sn Mole Fraction", xaxis_title="X (μm)", yaxis_title="Y (μm)", coloraxis_colorbar_title="Sn Mole Fraction")
    st.plotly_chart(fig)

# Sigmoidal interpolation
def h(c):
    return 1 / (1 + np.exp(-beta_sigmoid * (c - c0)))

# Free energy functions (non-dimensional)
def f_chemical(c):
    G_Li2Sn5 = -2.95e4 / W
    G_Sn = -2.47e4 / W
    return h(c) * G_Li2Sn5 + (1 - h(c)) * G_Sn + c * (1 - c)

def f_electromigration(c, phi):
    return (NA * e / W) * (h(c) * Z_Li2Sn5 * (1 - c) + (1 - h(c)) * Z_Sn * (1 - c)) * phi

def f_elastic(c):
    epsilon_xx = beta_2D * (1 - c)
    return C * (epsilon_xx**2)

# Derivatives
def df_chemical_dc(c):
    dh_dc = beta_sigmoid * h(c) * (1 - h(c))
    G_Li2Sn5 = -2.95e4 / W
    G_Sn = -2.47e4 / W
    return dh_dc * (G_Li2Sn5 - G_Sn) + (1 - 2 * c)

def df_electromigration_dc(c, phi):
    dh_dc = beta_sigmoid * h(c) * (1 - h(c))
    return (NA * e / W) * (dh_dc * (Z_Li2Sn5 - Z_Sn) * (1 - c) - (h(c) * Z_Li2Sn5 + (1 - h(c)) * Z_Sn)) * phi

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

# CFL condition
def compute_dt(c, dx, dy):
    M_max = np.max(M(c))
    dt = 0.5 * (dx**2 * dy**2) / (2 * M_max * kappa * (4 / dx**2 + 4 / dy**2))
    logger.info(f"Non-dimensional CFL time step: {dt:.1e} (dimensional: {dt*tau:.1e} s)")
    st.write(f"Initial CFL Time Step: {dt*tau:.1e} s")
    return dt

# Solve electric potential with Dirichlet BCs
def solve_potential(c):
    logger.info("Solving electric potential...")
    kappa = 1 / (h(c) * rho_Li2Sn5_star + (1 - h(c)) * rho_Sn_star)
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
                b[idx] = 0.04 / phi_0  # Dirichlet: 0.04 V
            elif j == 0:  # Insulating (y = 0)
                A[idx, idx] = 1
                if j + 1 < Ny:
                    A[idx, idx + Nx] = -1
            elif j == Ny - 1:  # Insulating (y = Ly)
                A[idx, idx] = 1
                if j - 1 >= 0:
                    A[idx, idx - Nx] = -1
            else:  # Interior
                A[idx, idx] = -2 * kappa[j, i] * (1 / dx**2 + 1 / dy**2)
                if i - 1 >= 0:
                    A[idx, idx - 1] = kappa[j, i] / dx**2
                if i + 1 < Nx:
                    A[idx, idx + 1] = kappa[j, i] / dx**2
                if j - 1 >= 0:
                    A[idx, idx - Nx] = kappa[j, i] / dy**2
                if j + 1 < Ny:
                    A[idx, idx + Nx] = kappa[j, i] / dy**2
    from scipy.sparse import csr_matrix
    A = csr_matrix(A)
    phi = spsolve(A, b).reshape((Ny, Nx))
    return np.maximum(phi, 0)  # Ensure positive potential

# Cahn-Hilliard RHS
def cahn_hilliard(t, c_flat):
    c = c_flat.reshape((Ny, Nx))
    phi = solve_potential(c)
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
    epsilon_yy = beta_2D * (1 - c)  # Isotropic expansion
    sigma_xx = C_dim * epsilon_xx
    sigma_yy = C_dim * epsilon_yy
    sigma_vm = np.abs(sigma_xx)  # Von Mises stress, since sigma_xx = sigma_yy, sigma_xy = 0
    laplacian_phi = laplacian_2d(phi, dx, dy) * (phi_0 / Lx**2)  # Dimensional Laplacian (V/m^2)
    return {
        "c": c,
        "phi": phi * phi_0,
        "epsilon_xx": epsilon_xx,
        "epsilon_yy": epsilon_yy,
        "sigma_xx": sigma_xx,
        "sigma_yy": sigma_yy,
        "sigma_vm": sigma_vm,
        "laplacian_phi": laplacian_phi
    }

# Visualization functions
def plot_plotly(data, title, label, x_scale=Lx, y_scale=Ly):
    fig = go.Figure(data=go.Heatmap(z=data, x=x*x_scale*1e6, y=y*y_scale*1e6, colorscale="Viridis"))
    fig.update_layout(title=title, xaxis_title="X (μm)", yaxis_title="Y (μm)", coloraxis_colorbar_title=label)
    st.plotly_chart(fig)

def plot_matplotlib(data, title, label, filename, x_scale=Lx, y_scale=Ly):
    plt.figure()
    plt.imshow(data, extent=[0, x_scale*1e6, 0, y_scale*1e6], origin="lower", cmap="viridis")
    plt.colorbar(label=label)
    plt.title(title)
    plt.xlabel("X (μm)")
    plt.ylabel("Y (μm)")
    plt.savefig(filename)
    st.image(filename)
    plt.close()

# Save to VTS (Structured Grid)
def save_vts(c, phi, epsilon_xx, epsilon_yy, sigma_xx, sigma_yy, sigma_vm, laplacian_phi, t, index):
    logger.info(f"Saving VTS file for t = {t*tau:.1e} s (index = {index})")
    points = np.zeros((Nx * Ny, 3))
    for j in range(Ny):
        for i in range(Nx):
            idx = j * Nx + i
            points[idx, 0] = x[i] * Lx
            points[idx, 1] = y[j] * Ly
            points[idx, 2] = 0.0
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (Nx, Ny, 1)
    grid["Concentration"] = c.T.flatten(order="F")
    grid["Potential"] = (phi * phi_0).T.flatten(order="F")
    grid["Strain_xx"] = epsilon_xx.T.flatten(order="F")
    grid["Strain_yy"] = epsilon_yy.T.flatten(order="F")
    grid["Stress_xx"] = sigma_xx.T.flatten(order="F")
    grid["Stress_yy"] = sigma_yy.T.flatten(order="F")
    grid["Von_Mises_Stress"] = sigma_vm.T.flatten(order="F")
    grid["Laplacian_Potential"] = laplacian_phi.T.flatten(order="F")
    grid.field_data["TIME"] = np.array([t * tau])
    grid.save(f"output_t_{index:04d}.vts", binary=True)

# Save to HDF5
def save_h5(c, phi, epsilon_xx, epsilon_yy, sigma_xx, sigma_yy, sigma_vm, laplacian_phi, t):
    logger.info(f"Saving HDF5 file for t = {t*tau:.1e} s")
    with h5py.File(f"output_t_{t*tau:.1e}.h5", "w") as f:
        f.create_dataset("Concentration", data=c)
        f.create_dataset("Potential", data=phi * phi_0)
        f.create_dataset("Strain_xx", data=epsilon_xx)
        f.create_dataset("Strain_yy", data=epsilon_yy)
        f.create_dataset("Stress_xx", data=sigma_xx)
        f.create_dataset("Stress_yy", data=sigma_yy)
        f.create_dataset("Von_Mises_Stress", data=sigma_vm)
        f.create_dataset("Laplacian_Potential", data=laplacian_phi)

# PyVista visualization with trame backend and Matplotlib fallback
def plot_pyvista(c, phi, epsilon_xx, epsilon_yy, sigma_xx, sigma_yy, sigma_vm, laplacian_phi, t):
    logger.info("Generating PyVista plot with trame backend...")
    try:
        grid = pv.RectilinearGrid(x*Lx, y*Ly)
        grid["Concentration"] = c.flatten(order="F")
        grid["Potential"] = (phi * phi_0).flatten(order="F")
        grid["Strain_xx"] = epsilon_xx.flatten(order="F")
        grid["Strain_yy"] = epsilon_yy.flatten(order="F")
        grid["Stress_xx"] = sigma_xx.flatten(order="F")
        grid["Stress_yy"] = sigma_yy.flatten(order="F")
        grid["Von_Mises_Stress"] = sigma_vm.flatten(order="F")
        grid["Laplacian_Potential"] = laplacian_phi.flatten(order="F")
        plotter = pv.Plotter(notebook=True)
        plotter.add_mesh(grid, scalars="Concentration", cmap="viridis")
        html_content = plotter._to_html()
        logger.info("PyVista plot generated successfully")
        st.write(f"PyVista Concentration Plot at t = {t*tau:.1e} s")
        st.components.v1.html(html_content, height=500)
    except Exception as e:
        logger.error(f"PyVista rendering failed: {str(e)}")
        st.warning(f"PyVista rendering failed: {str(e)}. Falling back to Matplotlib.")
        filename = f"pyvista_fallback_conc_t_{t*tau:.1e}.png"
        plt.figure()
        plt.imshow(c, extent=[0, Lx*1e6, 0, Ly*1e6], origin="lower", cmap="viridis")
        plt.colorbar(label="Sn Mole Fraction")
        plt.title(f"Concentration at t = {t*tau:.1e} s (Matplotlib Fallback)")
        plt.xlabel("X (μm)")
        plt.ylabel("Y (μm)")
        plt.savefig(filename)
        st.image(filename)
        plt.close()

# Run simulation
if st.button("Run Simulation"):
    min_t_max_dim = 1e-28
    if t_max_dim < min_t_max_dim:
        st.error(f"Simulation time (t_max = {t_max_dim:.1e} s) must be at least {min_t_max_dim:.1e} s.")
    else:
        log_output.clear()
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        log_placeholder = st.empty()
        
        def progress_callback(t, c_flat):
            progress = min(t / t_max, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Simulation progress: {progress*100:.1f}% (t = {t*tau:.1e} s)")
            with log_placeholder.container():
                st.text("\n".join(log_output[-10:]))
        
        with st.spinner("Running simulation..."):
            logger.info("Starting simulation...")
            c_flat = c.flatten()
            t_eval = np.linspace(t_start, t_max, num_t_eval)
            dt_initial = compute_dt(c, dx, dy)
            sol = solve_ivp(
                cahn_hilliard, (t_start, t_max), c_flat, t_eval=t_eval, method="BDF",
                atol=1e-6, rtol=1e-6, max_step=dt_initial
            )
            
            logger.info("Postprocessing results...")
            for i, t in enumerate(t_eval):
                c = sol.y[:, i].reshape((Ny, Nx))
                phi = solve_potential(c)
                vars = compute_variables(c, phi)
                
                st.subheader(f"Results at t = {t*tau:.1e} s")
                plot_plotly(vars["c"], f"Concentration at t = {t*tau:.1e} s", "Sn Mole Fraction")
                plot_matplotlib(vars["c"], f"Concentration at t = {t*tau:.1e} s", "Sn Mole Fraction", f"conc_t_{t*tau:.1e}.png")
                plot_plotly(vars["phi"], f"Potential at t = {t*tau:.1e} s", "Potential (V)")
                plot_matplotlib(vars["phi"], f"Potential at t = {t*tau:.1e} s", "Potential (V)", f"phi_t_{t*tau:.1e}.png")
                plot_plotly(vars["epsilon_xx"], f"Strain_xx at t = {t*tau:.1e} s", "Strain_xx")
                plot_matplotlib(vars["epsilon_xx"], f"Strain_xx at t = {t*tau:.1e} s", "Strain_xx", f"strain_xx_t_{t*tau:.1e}.png")
                plot_plotly(vars["epsilon_yy"], f"Strain_yy at t = {t*tau:.1e} s", "Strain_yy")
                plot_matplotlib(vars["epsilon_yy"], f"Strain_yy at t = {t*tau:.1e} s", "Strain_yy", f"strain_yy_t_{t*tau:.1e}.png")
                plot_plotly(vars["sigma_xx"], f"Stress_xx at t = {t*tau:.1e} s", "Stress_xx (Pa)")
                plot_matplotlib(vars["sigma_xx"], f"Stress_xx at t = {t*tau:.1e} s", "Stress_xx (Pa)", f"stress_xx_t_{t*tau:.1e}.png")
                plot_plotly(vars["sigma_yy"], f"Stress_yy at t = {t*tau:.1e} s", "Stress_yy (Pa)")
                plot_matplotlib(vars["sigma_yy"], f"Stress_yy at t = {t*tau:.1e} s", "Stress_yy (Pa)", f"stress_yy_t_{t*tau:.1e}.png")
                plot_plotly(vars["sigma_vm"], f"Von Mises Stress at t = {t*tau:.1e} s", "Von Mises Stress (Pa)")
                plot_matplotlib(vars["sigma_vm"], f"Von Mises Stress at t = {t*tau:.1e} s", "Von Mises Stress (Pa)", f"sigma_vm_t_{t*tau:.1e}.png")
                plot_plotly(vars["laplacian_phi"], f"Laplacian of Potential at t = {t*tau:.1e} s", "Laplacian Potential (V/m²)")
                plot_matplotlib(vars["laplacian_phi"], f"Laplacian of Potential at t = {t*tau:.1e} s", "Laplacian Potential (V/m²)", f"laplacian_phi_t_{t*tau:.1e}.png")
                
                save_vts(vars["c"], vars["phi"], vars["epsilon_xx"], vars["epsilon_yy"], vars["sigma_xx"], vars["sigma_yy"], vars["sigma_vm"], vars["laplacian_phi"], t, i)
                save_h5(vars["c"], vars["phi"], vars["epsilon_xx"], vars["epsilon_yy"], vars["sigma_xx"], vars["sigma_yy"], vars["sigma_vm"], vars["laplacian_phi"], t)
                plot_pyvista(vars["c"], vars["phi"], vars["epsilon_xx"], vars["epsilon_yy"], vars["sigma_xx"], vars["sigma_yy"], vars["sigma_vm"], vars["laplacian_phi"], t)
            
            progress_bar.progress(1.0)
            status_text.text("Simulation complete!")
            with log_placeholder.container():
                st.text("\n".join(log_output[-10:]))
            st.success("Simulation complete! Outputs saved as .vts and .h5 files.")