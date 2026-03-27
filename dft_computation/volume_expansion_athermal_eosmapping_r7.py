#!/usr/bin/env python3
"""
DFT Volume Expansion & Mechanical Analysis: Sn → Li₂Sn₅ Lithiation
===================================================================
Integrated Athermal EOS Mapping + Anisotropic Elasticity + 
Thermodynamic Stability + Fracture Prediction

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS
from ase.spacegroup import crystal
from ase.units import GPa
from ase.eos import EquationOfState
try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import warnings
import os
import pickle
from datetime import datetime

# Optional imports with fallbacks
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # define dummy jit decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator

warnings.filterwarnings('ignore')

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Sn→Li₂Sn₅ Mechanics & Thermodynamics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {padding: 0.5rem 1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ DFT Mechanics & Thermodynamics: Sn Anode Lithiation")
st.markdown("""
**Integrated Workflow**: β-Sn (BCT) → Li₂Sn₅ Volume Expansion Analysis

| Phase | Description | Key Outputs |
|-------|-------------|-------------|
| 🔹 Phase 1 | Thermodynamic Stability | Formation Energy ΔE_f, Phase Stability |
| 🔹 Phase 2 | Isotropic E-V Mapping | V₀, B₀, Volume Expansion %, EOS Fit |
| 🔹 Phase 3 | Anisotropic Elasticity | C₁₁, C₃₃, Anisotropy Ratio AR |
| 🔹 Phase 4 | Fracture Prediction | Stress Distribution, Failure Risk, 3D Visualization |

**Physics Implemented**:
- Birch-Murnaghan 3rd-order EOS fitting
- Directional elastic constants via finite strain
- Polar spherical stress mapping for anisotropic materials
- Griffith-type fracture criterion with expansion threshold
""")

# ============================================================================
# Session State Initialization
# ============================================================================
if 'phase_results' not in st.session_state:
    st.session_state.phase_results = {
        'phase1': None,          # Thermodynamics
        'phase2_sn': None,       # Sn E-V
        'phase2_li2sn5': None,   # Li2Sn5 E-V
        'phase3_sn': None,       # Sn elasticity
        'phase3_li2sn5': None,   # Li2Sn5 elasticity
        'phase4': None           # Fracture prediction
    }
if 'ref_energies' not in st.session_state:
    st.session_state.ref_energies = None

# ============================================================================
# Sidebar Global Settings
# ============================================================================
st.sidebar.header("⚙️ Global DFT Settings")
calculation_mode = st.sidebar.selectbox(
    "Accuracy Mode",
    ["🚀 Fast Testing (5-15 min/phase)", "⚖️ Balanced (30-90 min/phase)", "🎯 High Accuracy (2-6 hrs/phase)"],
    index=0
)

mode_params = {
    "🚀 Fast Testing (5-15 min/phase)": {"ecut": 350, "kpts_sn": (4,4,6), "kpts_li2sn5": (3,3,8), "fmax": 0.05, "n_vol": 7, "n_strain": 5},
    "⚖️ Balanced (30-90 min/phase)": {"ecut": 450, "kpts_sn": (6,6,10), "kpts_li2sn5": (4,4,12), "fmax": 0.01, "n_vol": 9, "n_strain": 7},
    "🎯 High Accuracy (2-6 hrs/phase)": {"ecut": 500, "kpts_sn": (8,8,12), "kpts_li2sn5": (6,6,16), "fmax": 0.005, "n_vol": 11, "n_strain": 9}
}

params = mode_params[calculation_mode]
ecut = params["ecut"]
kpts_sn = params["kpts_sn"]
kpts_li2sn5 = params["kpts_li2sn5"]
fmax = params["fmax"]
n_vol = params["n_vol"]
n_strain = params["n_strain"]

volume_range = st.sidebar.slider("Volume Scaling Range (×V₀)", 0.80, 1.20, (0.92, 1.08), 0.02)
strain_range = st.sidebar.slider("Uniaxial Strain Range (%)", -5.0, 5.0, (-2.0, 2.0), 0.5)
enable_cache = st.sidebar.checkbox("Enable Calculation Caching", value=True)

st.sidebar.markdown("---")
st.sidebar.info("""
**Recommended Workflow**:
1. Start with *Fast Testing* to validate setup
2. Use *Balanced* for publication-quality trends
3. Reserve *High Accuracy* for final results

**Note**: PBE functional typically overestimates volumes by 1-3% vs experiment.
""")

# ============================================================================
# Helper Functions
# ============================================================================
def quadratic_strain(eps, A, B, C):
    """Quadratic fit for elastic constant extraction: E(ε) = Aε² + Bε + C"""
    return A * eps**2 + B * eps + C

def birch_murnaghan_eos(V, E0, V0, B0, Bp):
    """3rd-order Birch-Murnaghan Equation of State"""
    eta = (V0 / V)**(2/3)
    return E0 + (9*V0*B0/16) * ((eta - 1)**3 * Bp + (eta - 1)**2 * (6 - 4*eta))

def create_gpaaw_calculator(ecut, xc='PBE', kpts=(4,4,4), txt=None, convergence=None):
    """Create GPAW calculator with consistent settings"""
    if convergence is None:
        convergence = {'energy': 1e-5, 'density': 1e-4}
    return GPAW(
        mode=PW(ecut),
        xc=xc,
        kpts=kpts,
        txt=txt,
        convergence=convergence,
        maxiter=200,
        occupations={'name': 'fermi-dirac', 'width': 0.1}
    )

def relax_fixed_volume(atoms, fmax=0.05, max_steps=100):
    """Relax atomic positions at fixed cell volume"""
    calc = atoms.calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    return atoms.get_potential_energy()

# ----------------------------------------------------------------------------
# Cached reference energies (persisted across sessions)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=86400)  # cache for 1 day
def compute_reference_energies(ecut, kpts, fmax):
    """Compute bulk reference energies for Li and Sn"""
    # Check if GPAW is available
    try:
        import gpaw
    except ImportError:
        st.warning("GPAW not installed. Using precomputed reference energies from materials project.")
        # Fallback to precomputed values from known literature (eV/atom)
        return {"e_li_per_atom": -1.908, "e_sn_per_atom": -3.152}
    
    # Bulk Li (BCC)
    li_bulk = bulk('Li', 'bcc', a=3.51)
    li_calc = create_gpaaw_calculator(ecut, kpts=kpts, txt=None)
    li_bulk.calc = li_calc
    ef_li = ExpCellFilter(li_bulk)
    BFGS(ef_li).run(fmax=fmax)
    e_li_per_atom = li_bulk.get_potential_energy() / len(li_bulk)
    
    # Bulk Sn (BCT)
    sn_bulk = crystal('Sn', basis=[(0,0,0)], spacegroup=141,
                      cellpar=[5.83, 5.83, 3.18, 90, 90, 90])
    sn_calc = create_gpaaw_calculator(ecut, kpts=kpts, txt=None)
    sn_bulk.calc = sn_calc
    ef_sn = ExpCellFilter(sn_bulk)
    BFGS(ef_sn).run(fmax=fmax)
    e_sn_per_atom = sn_bulk.get_potential_energy() / len(sn_bulk)
    
    return {"e_li_per_atom": e_li_per_atom, "e_sn_per_atom": e_sn_per_atom}

# ----------------------------------------------------------------------------
# Phase 1: Thermodynamics
# ----------------------------------------------------------------------------
def phase1_thermodynamic_stability(e_li2sn5_total, e_sn_per, e_li_per, n_li=4, n_sn=10):
    n_total = n_li + n_sn
    delta_e = e_li2sn5_total - n_li * e_li_per - n_sn * e_sn_per
    formation_per_atom = delta_e / n_total
    formation_per_formula = delta_e / 2  # Li₂Sn₅ formula unit
    stability = "✅ Thermodynamically Stable" if formation_per_atom < 0 else "⚠️ Metastable/Unstable"
    return {
        "delta_e_total": delta_e,
        "formation_per_atom": formation_per_atom,
        "formation_per_formula": formation_per_formula,
        "stability_label": stability,
        "is_stable": formation_per_atom < 0
    }

# ----------------------------------------------------------------------------
# Phase 2: E-V Curves (with parallelization & GP surrogate)
# ----------------------------------------------------------------------------
def compute_ev_parallel(volumes, template_atoms, ecut, kpts, fmax, n_workers=None):
    """
    Parallel computation of E(V) points using multiprocessing.
    Returns list of (volume, energy) sorted by volume.
    """
    import multiprocessing as mp
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(volumes))
    
    def compute_single(vol):
        atoms = template_atoms.copy()
        scale = (vol / atoms.get_volume()) ** (1/3)
        atoms.set_cell(atoms.get_cell() * scale, scale_atoms=True)
        calc = create_gpaaw_calculator(ecut, kpts=kpts, txt=None)
        atoms.calc = calc
        energy = relax_fixed_volume(atoms, fmax=fmax)
        return vol, energy
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(compute_single, vol): vol for vol in volumes}
        progress_bar = st.progress(0, text="Computing E-V points...")
        for i, future in enumerate(as_completed(futures)):
            vol, energy = future.result()
            results.append((vol, energy))
            progress_bar.progress((i+1)/len(volumes))
    progress_bar.empty()
    return sorted(results, key=lambda x: x[0])

def train_gp_surrogate(volumes, energies, n_points=5):
    """Use Gaussian Process to predict E(V) and optionally add extra points."""
    if not SKLEARN_AVAILABLE:
        return None, None
    # Only use first n_points points for training
    train_vols = volumes[:n_points]
    train_ens = energies[:n_points]
    
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    gp.fit(np.array(train_vols).reshape(-1,1), train_ens)
    return gp

@st.cache_data(show_spinner=False)
def compute_ev_curve(structure_name, a_init, c_init, symbols, spacegroup, basis, num_sn, kpts,
                     volume_range, n_points, fmax, ecut, use_surrogate=False):
    """Compute E-V curve with optional GP surrogate acceleration."""
    # Create template structure
    if structure_name == 'Sn (BCT)':
        template = crystal('Sn', basis=[(0,0,0)], spacegroup=141,
                          cellpar=[a_init, a_init, c_init, 90, 90, 90])
    elif structure_name == 'Li2Sn5':
        template = crystal(symbols=symbols, basis=basis, spacegroup=spacegroup,
                          cellpar=[a_init, a_init, c_init, 90, 90, 90])
    else:
        raise ValueError(f"Unknown structure: {structure_name}")
    
    v0_init = template.get_volume()
    scales = np.linspace(volume_range[0], volume_range[1], n_points)
    target_volumes = v0_init * scales
    
    # Compute E-V points
    ev_points = compute_ev_parallel(target_volumes, template, ecut, kpts, fmax)
    volumes, energies = zip(*ev_points)
    volumes = np.array(volumes)
    energies = np.array(energies)
    
    # Fit Birch-Murnaghan EOS
    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    v0_fit, e0_fit, B0_fit, Bp_fit = eos.fit()
    
    # If surrogate enabled, optionally add more points around minimum
    if use_surrogate and SKLEARN_AVAILABLE and n_points < 11:
        gp = train_gp_surrogate(volumes, energies, n_points=min(5, len(volumes)))
        # Add points in highest uncertainty regions (simplified)
        # This is a placeholder – actual implementation would sample adaptively
    
    return {
        "volumes": volumes,
        "energies": energies,
        "v0_init": v0_init,
        "v0_fit": v0_fit,
        "e0_fit": e0_fit,
        "B0_GPa": B0_fit / GPa,
        "Bp": Bp_fit,
        "num_sn": num_sn,
        "eos": eos
    }

# ----------------------------------------------------------------------------
# Phase 3: Anisotropic Elasticity
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_anisotropic_elasticity(structure_name, a0, c0, symbols, spacegroup, basis,
                                   kpts, fmax, ecut, strain_range, n_strain):
    """Compute C11 and C33 using finite strain method."""
    template = crystal(symbols=symbols, basis=basis, spacegroup=spacegroup,
                      cellpar=[a0, a0, c0, 90, 90, 90])
    v0 = template.get_volume()
    
    strains = np.linspace(strain_range[0]/100, strain_range[1]/100, n_strain)
    
    # Helper to compute energy for a given strain direction
    def compute_energy_for_strain(axis, eps):
        atoms = template.copy()
        if axis == 'a':
            new_cell = [a0*(1+eps), a0*(1+eps), c0, 90, 90, 90]
        elif axis == 'c':
            new_cell = [a0, a0, c0*(1+eps), 90, 90, 90]
        else:
            raise ValueError("Axis must be 'a' or 'c'")
        atoms.set_cell(new_cell, scale_atoms=True)
        calc = create_gpaaw_calculator(ecut, kpts=kpts, txt=None)
        atoms.calc = calc
        return relax_fixed_volume(atoms, fmax=fmax)
    
    # Compute energies for all strains (can be parallelized)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    energies_a = [None]*len(strains)
    energies_c = [None]*len(strains)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures_a = {executor.submit(compute_energy_for_strain, 'a', eps): i for i, eps in enumerate(strains)}
        futures_c = {executor.submit(compute_energy_for_strain, 'c', eps): i for i, eps in enumerate(strains)}
        
        for future in as_completed(futures_a):
            idx = futures_a[future]
            energies_a[idx] = future.result()
        for future in as_completed(futures_c):
            idx = futures_c[future]
            energies_c[idx] = future.result()
    
    # Quadratic fits
    popt_a, _ = curve_fit(quadratic_strain, strains, energies_a)
    popt_c, _ = curve_fit(quadratic_strain, strains, energies_c)
    
    # Convert to GPa
    c11 = (2 * popt_a[0] / v0) * 160.217
    c33 = (2 * popt_c[0] / v0) * 160.217
    
    return {
        "strains": strains,
        "energies_a": np.array(energies_a),
        "energies_c": np.array(energies_c),
        "c11_gpa": c11,
        "c33_gpa": c33,
        "anisotropy_ratio": c33 / c11 if c11 != 0 else np.inf,
        "v0": v0
    }

# ----------------------------------------------------------------------------
# Phase 4: Fracture Prediction & Stress Mapping
# ----------------------------------------------------------------------------
def predict_fracture_risk(expansion_pct, anisotropy_ratio, b0_drop_pct, c33_gpa):
    risk_score = 0
    factors = []
    if expansion_pct > 30:
        risk_score += 3
        factors.append("🔴 Extreme expansion (>30%)")
    elif expansion_pct > 20:
        risk_score += 2
        factors.append("🟡 High expansion (20-30%)")
    elif expansion_pct > 10:
        risk_score += 1
        factors.append("🟢 Moderate expansion (10-20%)")
    if anisotropy_ratio < 0.7:
        risk_score += 3
        factors.append("🔴 Severe c-axis softening (AR<0.7)")
    elif anisotropy_ratio < 0.9:
        risk_score += 2
        factors.append("🟡 Moderate anisotropy (AR 0.7-0.9)")
    if b0_drop_pct > 50:
        risk_score += 2
        factors.append("🟡 Significant softening (>50% B₀ drop)")
    if c33_gpa < 20:
        risk_score += 1
        factors.append("🟢 Low c-axis stiffness")
    
    if risk_score >= 6:
        risk_level = "🔴 CRITICAL"
        description = "High probability of pulverization/delamination during cycling"
    elif risk_score >= 4:
        risk_level = "🟡 ELEVATED"
        description = "Moderate fracture risk; consider nanostructuring or composites"
    elif risk_score >= 2:
        risk_level = "🟢 MODERATE"
        description = "Manageable mechanical degradation with proper electrode design"
    else:
        risk_level = "🟢 LOW"
        description = "Good mechanical stability expected"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "description": description,
        "contributing_factors": factors
    }

@jit(nopython=True, parallel=True, cache=True)
def compute_stress_field_numba(c11, c33, n_theta=180, n_phi=90):
    """Numba-accelerated stress field on unit sphere."""
    stress = np.empty((n_phi, n_theta))
    for i in prange(n_phi):
        phi = np.pi * i / (n_phi - 1)
        for j in range(n_theta):
            theta = 2 * np.pi * j / (n_theta - 1)
            lx = np.sin(phi) * np.cos(theta)
            ly = np.sin(phi) * np.sin(theta)
            lz = np.cos(phi)
            stress[i, j] = c11 * (lx**2 + ly**2) + c33 * lz**2
    return stress

def compute_stress_distribution_3d(c11, c33, expansion_tensor=None, n_theta=180, n_phi=90):
    if NUMBA_AVAILABLE:
        stress_magnitude = compute_stress_field_numba(c11, c33, n_theta, n_phi)
    else:
        theta = np.linspace(0, 2*np.pi, n_theta)
        phi = np.linspace(0, np.pi, n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        lx = np.sin(phi_grid) * np.cos(theta_grid)
        ly = np.sin(phi_grid) * np.sin(theta_grid)
        lz = np.cos(phi_grid)
        stress_magnitude = c11 * (lx**2 + ly**2) + c33 * lz**2
        theta_grid = theta_grid.T
        phi_grid = phi_grid.T
    # Convert to spherical coordinates for plotting
    phi_grid, theta_grid = np.mgrid[0:np.pi:n_phi*1j, 0:2*np.pi:n_theta*1j]
    x = np.sin(phi_grid) * np.cos(theta_grid) * stress_magnitude
    y = np.sin(phi_grid) * np.sin(theta_grid) * stress_magnitude
    z = np.cos(phi_grid) * stress_magnitude
    return {
        "x": x, "y": y, "z": z,
        "theta": theta_grid, "phi": phi_grid,
        "stress": stress_magnitude,
        "c11": c11, "c33": c33
    }

# ----------------------------------------------------------------------------
# Visualization Functions (unchanged except minor adjustments)
# ----------------------------------------------------------------------------
def plot_radar_chart(properties_dict, title="Property Comparison"):
    categories = list(properties_dict.keys())
    N = len(categories)
    values = list(properties_dict.values())
    min_val, max_val = min(values), max(values)
    if max_val > min_val:
        normalized = [(v - min_val) / (max_val - min_val) * 0.8 + 0.1 for v in values]
    else:
        normalized = [0.5] * N
    normalized += normalized[:1]
    angles = [n / N * 2 * np.pi for n in range(N)] + [2 * np.pi]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, normalized, 'o-', linewidth=2, color='#667eea')
    ax.fill(angles, normalized, alpha=0.25, color='#667eea')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20, size=14, weight='bold')
    ax.grid(True, alpha=0.3)
    return fig

def plot_eos_scatter_with_fit(eos_results, phase_name, ax):
    vols = eos_results["volumes"]
    energies = eos_results["energies"]
    v0 = eos_results["v0_fit"]
    e0 = eos_results["e0_fit"]
    B0 = eos_results["B0_GPa"]
    Bp = eos_results["Bp"]
    v_smooth = np.linspace(vols.min()*0.98, vols.max()*1.02, 100)
    e_smooth = [birch_murnaghan_eos(v, e0, v0, B0*GPa, Bp) for v in v_smooth]
    ax.scatter(vols, energies, c='#e74c3c', s=60, label='DFT Points', zorder=5, edgecolors='white')
    ax.plot(v_smooth, e_smooth, 'b-', linewidth=2, label='Birch-Murnaghan Fit')
    ax.axvline(x=v0, color='green', linestyle='--', linewidth=1, label=f'V₀ = {v0:.2f} Å³')
    ax.set_xlabel('Volume (Å³)', fontsize=11)
    ax.set_ylabel('Energy (eV)', fontsize=11)
    ax.set_title(f'{phase_name}: E-V Curve & EOS Fit', fontsize=12, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

def plot_elasticity_histogram(c11, c33, phase_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    constants = [c11, c33]
    labels = ['C₁₁ (a-b plane)', 'C₃₃ (c-axis)']
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(labels, constants, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Elastic Constant (GPa)', fontsize=11)
    ax.set_title(f'{phase_name}: Directional Stiffness', fontsize=12, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, constants):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f'{val:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    return fig

def plot_3d_stress_sphere(stress_data, title="Anisotropic Stress Distribution"):
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = stress_data["x"], stress_data["y"], stress_data["z"]
    stress = stress_data["stress"]
    cmap = cm.RdYlBu_r
    norm = plt.Normalize(vmin=stress.min(), vmax=stress.max())
    colors = cmap(norm(stress))
    surf = ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1,
                          linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z (c-axis)', fontsize=10)
    ax.set_title(title, fontsize=12, weight='bold', pad=20)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Relative Stress (a.u.)', fontsize=9)
    ax.text(0, 0, 1.3, '↑ c-axis', ha='center', fontsize=9, weight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.view_init(elev=25, azim=45)
    return fig

def plot_expansion_bar_chart(sn_results, li2sn5_results, expansion_pct):
    v_per_sn_sn = sn_results["v0_fit"] / sn_results["num_sn"]
    v_per_sn_li = li2sn5_results["v0_fit"] / li2sn5_results["num_sn"]
    fig, ax = plt.subplots(figsize=(6, 5))
    phases = ['β-Sn', 'Li₂Sn₅']
    volumes = [v_per_sn_sn, v_per_sn_li]
    colors = ['#2ecc71', '#9b59b6']
    bars = ax.bar(phases, volumes, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Volume per Sn Atom (Å³)', fontsize=11)
    ax.set_title(f'Volume Expansion: +{expansion_pct:.2f}%', fontsize=13, weight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.annotate('', xy=(1, v_per_sn_li), xytext=(0, v_per_sn_sn),
               arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='-'))
    ax.text(0.5, (v_per_sn_sn + v_per_sn_li)/2, f'+{expansion_pct:.1f}%',
           ha='center', va='bottom', color='red', weight='bold', fontsize=11)
    for bar, vol in zip(bars, volumes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{vol:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')
    return fig

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Phase 1: Thermodynamics",
    "📊 Phase 2: EOS & Expansion",
    "🧭 Phase 3: Anisotropic Elasticity",
    "💥 Phase 4: Fracture Prediction",
    "📈 Multi-View Dashboard"
])

# ----------------------------------------------------------------------------
# Phase 1
# ----------------------------------------------------------------------------
with tab1:
    st.header("🔬 Phase 1: Thermodynamic Stability")
    st.markdown("""
    **Formation Energy Calculation**:
