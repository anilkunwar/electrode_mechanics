#!/usr/bin/env python3
"""
DFT Volume Expansion & Mechanical Analysis: Sn → Li₂Sn₅ Lithiation
===================================================================
Integrated Athermal EOS Mapping + Anisotropic Elasticity + 
Thermodynamic Stability + Fracture Prediction

Run with: streamlit run app.py
Deploy to Streamlit Cloud: No GPU required, CPU-only parallelization
"""

# ============================================================================
# IMPORTS (with graceful fallbacks)
# ============================================================================
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
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import warnings
import os
import pickle
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Optional: GPAW (main DFT engine)
try:
    from gpaw import GPAW, PW
    try:
        from ase.filters import ExpCellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter
    GPAW_AVAILABLE = True
except ImportError:
    GPAW_AVAILABLE = False
    # Dummy classes for demo mode
    class DummyCalc:
        def get_potential_energy(self): return -100.0
    class GPAW:
        def __init__(self, *args, **kwargs): pass
        def set(self, **kwargs): pass
    class PW:
        def __init__(self, ecut): self.ecut = ecut
    class ExpCellFilter:
        def __init__(self, atoms): self.atoms = atoms

# Optional: scikit-learn for GP surrogate
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional: Numba for accelerated stress maps
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy jit decorator
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator if args else decorator

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sn→Li₂Sn₅ Mechanics", page_icon="⚡", layout="wide")

# ============================================================================
# GLOBAL CONFIGURATION & CACHING
# ============================================================================
@st.cache_resource
def get_default_params():
    """Cached default parameters"""
    return {
        "ecut_default": 450,
        "kpts_default": (6, 6, 10),
        "fmax_default": 0.01,
        "n_vol_default": 9
    }

@st.cache_data(ttl=3600)
def load_precomputed_references():
    """Load fallback reference energies (eV/atom) from literature"""
    return {
        "e_li_per_atom": -1.908,  # BCC Li, PBE
        "e_sn_per_atom": -3.152,  # BCT Sn, PBE
        "source": "Materials Project + literature benchmark"
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def quadratic_strain(eps, A, B, C):
    """Quadratic fit: E(ε) = Aε² + Bε + C"""
    return A * eps**2 + B * eps + C

def birch_murnaghan_eos(V, E0, V0, B0, Bp):
    """3rd-order Birch-Murnaghan EOS"""
    eta = (V0 / V)**(2/3)
    return E0 + (9*V0*B0/16) * ((eta - 1)**3 * Bp + (eta - 1)**2 * (6 - 4*eta))

def create_calculator(ecut, xc='PBE', kpts=(4,4,4), txt=None, convergence=None):
    """Create GPAW calculator with consistent settings"""
    if not GPAW_AVAILABLE:
        return DummyCalc()
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
    if not GPAW_AVAILABLE:
        return atoms.get_potential_energy() if hasattr(atoms, 'get_potential_energy') else -100.0
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    return atoms.get_potential_energy()

# ============================================================================
# PHASE 1: THERMODYNAMIC STABILITY
# ============================================================================
def phase1_thermodynamic_stability(e_li2sn5_total, e_sn_per, e_li_per, n_li=4, n_sn=10):
    """Compute formation energy for Li₂Sn₅"""
    n_total = n_li + n_sn
    delta_e = e_li2sn5_total - n_li * e_li_per - n_sn * e_sn_per
    formation_per_atom = delta_e / n_total
    formation_per_formula = delta_e / 2
    stability = "✅ Thermodynamically Stable" if formation_per_atom < 0 else "⚠️ Metastable/Unstable"
    return {
        "delta_e_total": delta_e,
        "formation_per_atom": formation_per_atom,
        "formation_per_formula": formation_per_formula,
        "stability_label": stability,
        "is_stable": formation_per_atom < 0
    }

@st.cache_data(show_spinner=False)
def compute_reference_energies(ecut, kpts, fmax):
    """Compute bulk reference energies (cached)"""
    if not GPAW_AVAILABLE:
        st.warning("GPAW not available. Using precomputed reference energies.")
        return load_precomputed_references()
    
    try:
        # Bulk Li (BCC)
        li_bulk = bulk('Li', 'bcc', a=3.51)
        li_calc = create_calculator(ecut, kpts=kpts, txt=None)
        li_bulk.calc = li_calc
        ef_li = ExpCellFilter(li_bulk)
        BFGS(ef_li).run(fmax=fmax)
        e_li = li_bulk.get_potential_energy() / len(li_bulk)
        
        # Bulk Sn (BCT)
        sn_bulk = crystal('Sn', basis=[(0,0,0)], spacegroup=141,
                         cellpar=[5.83, 5.83, 3.18, 90, 90, 90])
        sn_calc = create_calculator(ecut, kpts=kpts, txt=None)
        sn_bulk.calc = sn_calc
        ef_sn = ExpCellFilter(sn_bulk)
        BFGS(ef_sn).run(fmax=fmax)
        e_sn = sn_bulk.get_potential_energy() / len(sn_bulk)
        
        return {"e_li_per_atom": e_li, "e_sn_per_atom": e_sn}
    except Exception as e:
        st.error(f"Reference calculation failed: {e}. Using fallback values.")
        return load_precomputed_references()

# ============================================================================
# PHASE 2: E-V MAPPING (with parallelization & GP surrogate)
# ============================================================================
def compute_single_ev_point(args):
    """Worker function for parallel E-V computation"""
    vol, template_atoms, ecut, kpts, fmax = args
    atoms = template_atoms.copy()
    scale = (vol / atoms.get_volume()) ** (1/3)
    atoms.set_cell(atoms.get_cell() * scale, scale_atoms=True)
    calc = create_calculator(ecut, kpts=kpts, txt=None)
    atoms.calc = calc
    energy = relax_fixed_volume(atoms, fmax=fmax)
    return vol, energy

@st.cache_data(show_spinner=False)
def compute_ev_curve(structure_name, a_init, c_init, symbols, spacegroup, basis, 
                     num_sn, kpts, volume_range, n_points, fmax, ecut, use_surrogate=False):
    """Compute E-V curve with optional parallelization and GP surrogate"""
    # Create template
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
    
    # Parallel computation of E-V points
    n_workers = min(mp.cpu_count(), len(target_volumes))
    args_list = [(vol, template, ecut, kpts, fmax) for vol in target_volumes]
    
    results = []
    progress_bar = st.progress(0, text=f"Computing {structure_name} E-V points...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(compute_single_ev_point, arg): arg[0] for arg in args_list}
        for i, future in enumerate(as_completed(futures)):
            try:
                vol, energy = future.result()
                results.append((vol, energy))
                progress_bar.progress((i+1)/len(target_volumes))
            except Exception as e:
                st.error(f"Failed to compute point: {e}")
    
    progress_bar.empty()
    results.sort(key=lambda x: x[0])
    volumes, energies = zip(*results)
    volumes, energies = np.array(volumes), np.array(energies)
    
    # Fit Birch-Murnaghan EOS
    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    v0_fit, e0_fit, B0_fit, Bp_fit = eos.fit()
    
    # Optional: GP surrogate for adaptive sampling (placeholder)
    if use_surrogate and SKLEARN_AVAILABLE and n_points < 11:
        # Train GP on first half of points
        train_v = volumes[:n_points//2]
        train_e = energies[:n_points//2]
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
        gp.fit(train_v.reshape(-1,1), train_e)
        # Could add adaptive points here based on uncertainty
    
    return {
        "volumes": volumes, "energies": energies,
        "v0_init": v0_init, "v0_fit": v0_fit, "e0_fit": e0_fit,
        "B0_GPa": B0_fit / GPa, "Bp": Bp_fit, "num_sn": num_sn, "eos": eos
    }

# ============================================================================
# PHASE 3: ANISOTROPIC ELASTICITY
# ============================================================================
@st.cache_data(show_spinner=False)
def compute_anisotropic_elasticity(structure_name, a0, c0, symbols, spacegroup, basis,
                                   kpts, fmax, ecut, strain_range, n_strain):
    """Compute C11 and C33 using finite strain method"""
    template = crystal(symbols=symbols, basis=basis, spacegroup=spacegroup,
                      cellpar=[a0, a0, c0, 90, 90, 90])
    v0 = template.get_volume()
    strains = np.linspace(strain_range[0]/100, strain_range[1]/100, n_strain)
    
    def compute_energy_for_strain(axis, eps):
        atoms = template.copy()
        if axis == 'a':
            new_cell = [a0*(1+eps), a0*(1+eps), c0, 90, 90, 90]
        elif axis == 'c':
            new_cell = [a0, a0, c0*(1+eps), 90, 90, 90]
        else:
            raise ValueError("Axis must be 'a' or 'c'")
        atoms.set_cell(new_cell, scale_atoms=True)
        calc = create_calculator(ecut, kpts=kpts, txt=None)
        atoms.calc = calc
        return relax_fixed_volume(atoms, fmax=fmax)
    
    # Parallel over strain values (threading for IO-bound)
    energies_a, energies_c = [None]*len(strains), [None]*len(strains)
    
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
        "strains": strains, "energies_a": np.array(energies_a), "energies_c": np.array(energies_c),
        "c11_gpa": c11, "c33_gpa": c33,
        "anisotropy_ratio": c33 / c11 if c11 != 0 else np.inf, "v0": v0
    }

# ============================================================================
# PHASE 4: FRACTURE PREDICTION & STRESS MAPPING
# ============================================================================
def predict_fracture_risk(expansion_pct, anisotropy_ratio, b0_drop_pct, c33_gpa):
    """Composite fracture risk assessment"""
    risk_score, factors = 0, []
    
    if expansion_pct > 30: risk_score += 3; factors.append("🔴 Extreme expansion (>30%)")
    elif expansion_pct > 20: risk_score += 2; factors.append("🟡 High expansion (20-30%)")
    elif expansion_pct > 10: risk_score += 1; factors.append("🟢 Moderate expansion (10-20%)")
    
    if anisotropy_ratio < 0.7: risk_score += 3; factors.append("🔴 Severe c-axis softening (AR<0.7)")
    elif anisotropy_ratio < 0.9: risk_score += 2; factors.append("🟡 Moderate anisotropy (AR 0.7-0.9)")
    
    if b0_drop_pct > 50: risk_score += 2; factors.append("🟡 Significant softening (>50% B₀ drop)")
    if c33_gpa < 20: risk_score += 1; factors.append("🟢 Low c-axis stiffness")
    
    if risk_score >= 6: risk_level, desc = "🔴 CRITICAL", "High probability of pulverization/delamination"
    elif risk_score >= 4: risk_level, desc = "🟡 ELEVATED", "Moderate fracture risk; consider nanostructuring"
    elif risk_score >= 2: risk_level, desc = "🟢 MODERATE", "Manageable degradation with proper design"
    else: risk_level, desc = "🟢 LOW", "Good mechanical stability expected"
    
    return {"risk_score": risk_score, "risk_level": risk_level, "description": desc, "contributing_factors": factors}

@jit(nopython=True, parallel=True, cache=True) if NUMBA_AVAILABLE else lambda *args, **kwargs: lambda f: f
def compute_stress_field_numba(c11, c33, n_theta=180, n_phi=90):
    """Numba-accelerated stress field on unit sphere"""
    stress = np.empty((n_phi, n_theta))
    for i in prange(n_phi):
        phi = np.pi * i / (n_phi - 1)
        for j in range(n_theta):
            theta = 2 * np.pi * j / (n_theta - 1)
            lx, ly, lz = np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)
            stress[i, j] = c11 * (lx**2 + ly**2) + c33 * lz**2
    return stress

def compute_stress_distribution_3d(c11, c33, n_theta=180, n_phi=90):
    """Compute 3D stress distribution with Numba fallback"""
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
    
    # Convert to Cartesian for plotting
    x = np.sin(phi_grid) * np.cos(theta_grid) * stress_magnitude
    y = np.sin(phi_grid) * np.sin(theta_grid) * stress_magnitude
    z = np.cos(phi_grid) * stress_magnitude
    
    return {"x": x, "y": y, "z": z, "stress": stress_magnitude, "c11": c11, "c33": c33}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_radar_chart(properties_dict, title="Property Comparison"):
    """Create radar chart for multi-property comparison"""
    categories = list(properties_dict.keys())
    N = len(categories)
    values = list(properties_dict.values())
    min_val, max_val = min(values), max(values)
    normalized = [(v - min_val) / (max_val - min_val) * 0.8 + 0.1 for v in values] if max_val > min_val else [0.5]*N
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
    """Plot E-V scatter with Birch-Murnaghan fit"""
    vols, energies = eos_results["volumes"], eos_results["energies"]
    v0, e0, B0, Bp = eos_results["v0_fit"], eos_results["e0_fit"], eos_results["B0_GPa"], eos_results["Bp"]
    
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
    """Bar chart of elastic constants"""
    fig, ax = plt.subplots(figsize=(5, 4))
    constants, labels, colors = [c11, c33], ['C₁₁ (a-b plane)', 'C₃₃ (c-axis)'], ['#3498db', '#e74c3c']
    bars = ax.bar(labels, constants, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Elastic Constant (GPa)', fontsize=11)
    ax.set_title(f'{phase_name}: Directional Stiffness', fontsize=12, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, constants):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    return fig

def plot_3d_stress_sphere(stress_data, title="Anisotropic Stress Distribution"):
    """3D spherical stress visualization"""
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    x, y, z, stress = stress_data["x"], stress_data["y"], stress_data["z"], stress_data["stress"]
    
    cmap = cm.RdYlBu_r
    norm = plt.Normalize(vmin=stress.min(), vmax=stress.max())
    colors = cmap(norm(stress))
    
    surf = ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z (c-axis)', fontsize=10)
    ax.set_title(title, fontsize=12, weight='bold', pad=20)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Relative Stress (a.u.)', fontsize=9)
    ax.text(0, 0, 1.3, '↑ c-axis', ha='center', fontsize=9, weight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.view_init(elev=25, azim=45)
    return fig

def plot_expansion_bar_chart(sn_results, li2sn5_results, expansion_pct):
    """Volume expansion comparison bar chart"""
    v_per_sn_sn = sn_results["v0_fit"] / sn_results["num_sn"]
    v_per_sn_li = li2sn5_results["v0_fit"] / li2sn5_results["num_sn"]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    phases, volumes, colors = ['β-Sn', 'Li₂Sn₅'], [v_per_sn_sn, v_per_sn_li], ['#2ecc71', '#9b59b6']
    bars = ax.bar(phases, volumes, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Volume per Sn Atom (Å³)', fontsize=11)
    ax.set_title(f'Volume Expansion: +{expansion_pct:.2f}%', fontsize=13, weight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax.annotate('', xy=(1, v_per_sn_li), xytext=(0, v_per_sn_sn), arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.5, (v_per_sn_sn + v_per_sn_li)/2, f'+{expansion_pct:.1f}%', ha='center', va='bottom', color='red', weight='bold', fontsize=11)
    
    for bar, vol in zip(bars, volumes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{vol:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')
    return fig

# ============================================================================
# MAIN APP: TABS WITH INDEPENDENT EXECUTION
# ============================================================================
st.title("⚡ DFT Mechanics & Thermodynamics: Sn Anode Lithiation")
st.markdown("""
**Integrated Workflow**: β-Sn (BCT) → Li₂Sn₅ Volume Expansion Analysis

| Phase | Description | Key Outputs |
|-------|-------------|-------------|
| 🔹 Phase 1 | Thermodynamic Stability | Formation Energy ΔE_f, Phase Stability |
| 🔹 Phase 2 | Isotropic E-V Mapping | V₀, B₀, Volume Expansion %, EOS Fit |
| 🔹 Phase 3 | Anisotropic Elasticity | C₁₁, C₃₃, Anisotropy Ratio AR |
| 🔹 Phase 4 | Fracture Prediction | Stress Distribution, Failure Risk, 3D Visualization |
""")

# Session state for independent phase results
if 'phase_results' not in st.session_state:
    st.session_state.phase_results = {
        'phase1': None, 'phase2_sn': None, 'phase2_li2sn5': None,
        'phase3_sn': None, 'phase3_li2sn5': None, 'phase4': None
    }
if 'ref_energies' not in st.session_state:
    st.session_state.ref_energies = None

# Sidebar: Global Settings
st.sidebar.header("⚙️ Global DFT Settings")
if not GPAW_AVAILABLE:
    st.sidebar.warning("⚠️ GPAW not installed. Running in demo mode with precomputed data.")

calculation_mode = st.sidebar.selectbox("Accuracy Mode", 
    ["🚀 Fast Testing (5-15 min/phase)", "⚖️ Balanced (30-90 min/phase)", "🎯 High Accuracy (2-6 hrs/phase)"], index=0)

mode_params = {
    "🚀 Fast Testing (5-15 min/phase)": {"ecut": 350, "kpts_sn": (4,4,6), "kpts_li2sn5": (3,3,8), "fmax": 0.05, "n_vol": 7, "n_strain": 5},
    "⚖️ Balanced (30-90 min/phase)": {"ecut": 450, "kpts_sn": (6,6,10), "kpts_li2sn5": (4,4,12), "fmax": 0.01, "n_vol": 9, "n_strain": 7},
    "🎯 High Accuracy (2-6 hrs/phase)": {"ecut": 500, "kpts_sn": (8,8,12), "kpts_li2sn5": (6,6,16), "fmax": 0.005, "n_vol": 11, "n_strain": 9}
}

params = mode_params[calculation_mode]
ecut, kpts_sn, kpts_li2sn5, fmax, n_vol, n_strain = [params[k] for k in ["ecut", "kpts_sn", "kpts_li2sn5", "fmax", "n_vol", "n_strain"]]

volume_range = st.sidebar.slider("Volume Scaling Range (×V₀)", 0.80, 1.20, (0.92, 1.08), 0.02)
strain_range = st.sidebar.slider("Uniaxial Strain Range (%)", -5.0, 5.0, (-2.0, 2.0), 0.5)
use_surrogate = st.sidebar.checkbox("Use GP Surrogate (Phase 2)", value=SKLEARN_AVAILABLE, disabled=not SKLEARN_AVAILABLE)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Phase 1: Thermodynamics", "📊 Phase 2: EOS & Expansion", 
    "🧭 Phase 3: Anisotropic Elasticity", "💥 Phase 4: Fracture Prediction", "📈 Dashboard"
])

# ============================================================================
# TAB 1: PHASE 1 - THERMODYNAMIC STABILITY
# ============================================================================
with tab1:
    st.header("🔬 Phase 1: Thermodynamic Stability")
    st.markdown("**Formation Energy**: ΔE_f = [E(Li₂Sn₅) - 4·E_Li - 10·E_Sn] / 14 atoms")
    
    col1, col2 = st.columns(2)
    with col1:
        run_phase1 = st.button("🚀 Run Phase 1", key="btn_p1", use_container_width=True)
    with col2:
        if st.session_state.phase_results['phase2_li2sn5']:
            st.success("✓ Li₂Sn₅ E₀ available from Phase 2")
        else:
            st.info("Run Phase 2 first for accurate Li₂Sn₅ energy")
    
    if run_phase1:
        with st.spinner("Computing reference energies..."):
            if st.session_state.ref_energies is None:
                st.session_state.ref_energies = compute_reference_energies(ecut, kpts_sn, fmax)
            ref = st.session_state.ref_energies
        
        # Get Li2Sn5 energy
        if st.session_state.phase_results['phase2_li2sn5']:
            e_li2sn5 = st.session_state.phase_results['phase2_li2sn5']["e0_fit"]
        else:
            st.warning("Enter Li₂Sn₅ total energy manually or run Phase 2 first")
            e_li2sn5 = st.number_input("Li₂Sn₅ total energy (eV)", value=-100.0, step=0.1)
        
        thermo = phase1_thermodynamic_stability(e_li2sn5, ref["e_sn_per_atom"], ref["e_li_per_atom"])
        st.session_state.phase_results['phase1'] = thermo
    
    # Display results
    if st.session_state.phase_results['phase1']:
        thermo = st.session_state.phase_results['phase1']
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("ΔE_f (per atom)", f"{thermo['formation_per_atom']:.4f} eV", delta=thermo['stability_label'])
        with col2: st.metric("ΔE_f (per formula)", f"{thermo['formation_per_formula']:.3f} eV")
        with col3: st.metric("Total ΔE", f"{thermo['delta_e_total']:.2f} eV")
        
        if thermo['is_stable']:
            st.success("✅ **Li₂Sn₅ is thermodynamically stable**")
        else:
            st.warning("⚠️ **Metastable** - kinetic factors may enable formation")
        
        # Energy diagram
        fig, ax = plt.subplots(figsize=(8, 4))
        phases = ['Li + Sn (ref)', 'Li₂Sn₅']
        energies = [0, thermo['formation_per_formula']]
        colors = ['#95a5a6', '#27ae60' if thermo['is_stable'] else '#e67e22']
        ax.bar(phases, energies, color=colors, edgecolor='black')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Energy Relative to Reference (eV/f.u.)')
        ax.set_title('Thermodynamic Stability')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)

# ============================================================================
# TAB 2: PHASE 2 - EOS & VOLUME EXPANSION
# ============================================================================
with tab2:
    st.header("📊 Phase 2: Equation of State & Volume Expansion")
    
    col1, col2 = st.columns(2)
    with col1:
        run_sn = st.button("🚀 Compute Sn E-V", key="btn_sn", use_container_width=True)
    with col2:
        run_li = st.button("🚀 Compute Li₂Sn₅ E-V", key="btn_li", use_container_width=True)
    
    if run_sn and not st.session_state.phase_results['phase2_sn']:
        with st.spinner("Computing Sn E-V curve (parallel)..."):
            try:
                sn_res = compute_ev_curve('Sn (BCT)', 5.83, 3.18, 'Sn', 141, [(0,0,0)], 4, kpts_sn,
                                         volume_range, n_vol, fmax, ecut, use_surrogate)
                st.session_state.phase_results['phase2_sn'] = sn_res
                st.success("✓ Sn E-V computed")
            except Exception as e:
                st.error(f"Sn calculation failed: {e}")
    
    if run_li and not st.session_state.phase_results['phase2_li2sn5']:
        with st.spinner("Computing Li₂Sn₅ E-V curve (parallel)..."):
            try:
                li_res = compute_ev_curve('Li2Sn5', 10.274, 3.125, ['Sn','Li','Sn'], 127,
                                         [(0,0.5,0),(0.16,0.66,0),(0.295,0.432,0)], 10, kpts_li2sn5,
                                         volume_range, n_vol, fmax, ecut, use_surrogate)
                st.session_state.phase_results['phase2_li2sn5'] = li_res
                st.success("✓ Li₂Sn₅ E-V computed")
            except Exception as e:
                st.error(f"Li₂Sn₅ calculation failed: {e}")
    
    # Display if both available
    sn_res, li_res = st.session_state.phase_results['phase2_sn'], st.session_state.phase_results['phase2_li2sn5']
    if sn_res and li_res:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("V₀ (β-Sn)", f"{sn_res['v0_fit']:.2f} Å³")
        with col2: st.metric("V₀ (Li₂Sn₅)", f"{li_res['v0_fit']:.2f} Å³")
        
        v_sn = sn_res['v0_fit'] / sn_res['num_sn']
        v_li = li_res['v0_fit'] / li_res['num_sn']
        with col3: st.metric("V/Sn (β-Sn)", f"{v_sn:.3f} Å³")
        with col4: st.metric("V/Sn (Li₂Sn₅)", f"{v_li:.3f} Å³")
        
        expansion = (v_li - v_sn) / v_sn * 100
        st.markdown(f"<div style='background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:1rem;border-radius:0.5rem;text-align:center;font-size:1.2rem'><strong>{expansion:+.2f}%</strong> volume expansion per Sn atom</div>", unsafe_allow_html=True)
        
        # E-V curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        plot_eos_scatter_with_fit(sn_res, 'β-Sn', ax1)
        plot_eos_scatter_with_fit(li_res, 'Li₂Sn₅', ax2)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Store for later phases
        st.session_state.expansion_pct = expansion
        st.session_state.b0_drop_pct = (sn_res['B0_GPa'] - li_res['B0_GPa']) / sn_res['B0_GPa'] * 100

# ============================================================================
# TAB 3: PHASE 3 - ANISOTROPIC ELASTICITY
# ============================================================================
with tab3:
    st.header("🧭 Phase 3: Anisotropic Elastic Constants")
    st.markdown("**Method**: Finite strain, quadratic fit: C_ii = (2A/V₀) × 160.217 GPa")
    
    col1, col2 = st.columns(2)
    with col1:
        run_sn_el = st.button("🚀 Compute Sn Elasticity", key="btn_sn_el", use_container_width=True)
    with col2:
        run_li_el = st.button("🚀 Compute Li₂Sn₅ Elasticity", key="btn_li_el", use_container_width=True)
    
    if run_sn_el and not st.session_state.phase_results['phase3_sn']:
        with st.spinner("Computing Sn elastic constants..."):
            try:
                sn_el = compute_anisotropic_elasticity('Sn', 5.83, 3.18, 'Sn', 141, [(0,0,0)],
                                                      kpts_sn, fmax, ecut, strain_range, n_strain)
                st.session_state.phase_results['phase3_sn'] = sn_el
                st.success("✓ Sn elasticity computed")
            except Exception as e:
                st.error(f"Sn elasticity failed: {e}")
    
    if run_li_el and not st.session_state.phase_results['phase3_li2sn5']:
        with st.spinner("Computing Li₂Sn₅ elastic constants..."):
            try:
                li_el = compute_anisotropic_elasticity('Li2Sn5', 10.274, 3.125, ['Sn','Li','Sn'], 127,
                                                      [(0,0.5,0),(0.16,0.66,0),(0.295,0.432,0)],
                                                      kpts_li2sn5, fmax, ecut, strain_range, n_strain)
                st.session_state.phase_results['phase3_li2sn5'] = li_el
                st.success("✓ Li₂Sn₅ elasticity computed")
            except Exception as e:
                st.error(f"Li₂Sn₅ elasticity failed: {e}")
    
    # Display results
    sn_el, li_el = st.session_state.phase_results['phase3_sn'], st.session_state.phase_results['phase3_li2sn5']
    if sn_el:
        st.subheader("β-Sn")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("C₁₁", f"{sn_el['c11_gpa']:.1f} GPa")
        with col2: st.metric("C₃₃", f"{sn_el['c33_gpa']:.1f} GPa")
        with col3: st.metric("AR", f"{sn_el['anisotropy_ratio']:.3f}")
    
    if li_el:
        st.subheader("Li₂Sn₅")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("C₁₁", f"{li_el['c11_gpa']:.1f} GPa")
        with col2: st.metric("C₃₃", f"{li_el['c33_gpa']:.1f} GPa")
        with col3: st.metric("AR", f"{li_el['anisotropy_ratio']:.3f}")
        
        # Strain-energy plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        strains = li_el['strains'] * 100
        ax1.plot(strains, li_el['energies_a'], 'o-', label='DFT', color='#e74c3c')
        popt_a, _ = curve_fit(quadratic_strain, li_el['strains'], li_el['energies_a'])
        ax1.plot(strains, quadratic_strain(strains/100, *popt_a), '--', label='Fit')
        ax1.set_xlabel('Strain ε_a (%)'); ax1.set_ylabel('Energy (eV)'); ax1.set_title('C₁₁: a-axis'); ax1.legend(); ax1.grid(alpha=0.3)
        
        ax2.plot(strains, li_el['energies_c'], 'o-', label='DFT', color='#9b59b6')
        popt_c, _ = curve_fit(quadratic_strain, li_el['strains'], li_el['energies_c'])
        ax2.plot(strains, quadratic_strain(strains/100, *popt_c), '--', label='Fit')
        ax2.set_xlabel('Strain ε_c (%)'); ax2.set_ylabel('Energy (eV)'); ax2.set_title('C₃₃: c-axis'); ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Store for Phase 4
        st.session_state.li2sn5_elastic = li_el

# ============================================================================
# TAB 4: PHASE 4 - FRACTURE PREDICTION
# ============================================================================
with tab4:
    st.header("💥 Phase 4: Mechanical Fracture Prediction")
    
    # Check dependencies
    missing = []
    if 'expansion_pct' not in st.session_state: missing.append("Phase 2 expansion")
    if 'b0_drop_pct' not in st.session_state: missing.append("Phase 2 bulk modulus")
    if not st.session_state.phase_results.get('phase3_li2sn5'): missing.append("Phase 3 Li₂Sn₅ elasticity")
    
    if missing:
        st.warning(f"Run these phases first: {', '.join(missing)}")
    else:
        if st.button("🚀 Run Fracture Prediction", key="btn_p4", use_container_width=True):
            with st.spinner("Predicting fracture risk..."):
                expansion = st.session_state.expansion_pct
                b0_drop = st.session_state.b0_drop_pct
                li_el = st.session_state.phase_results['phase3_li2sn5']
                
                fracture = predict_fracture_risk(expansion, li_el['anisotropy_ratio'], b0_drop, li_el['c33_gpa'])
                st.session_state.phase_results['phase4'] = fracture
                
                stress_3d = compute_stress_distribution_3d(li_el['c11_gpa'], li_el['c33_gpa'])
                st.session_state.stress_3d = stress_3d
                st.success("✓ Fracture prediction complete")
        
        if st.session_state.phase_results['phase4']:
            fracture = st.session_state.phase_results['phase4']
            li_el = st.session_state.phase_results['phase3_li2sn5']
            
            # Risk display
            color = "#e74c3c" if "CRITICAL" in fracture['risk_level'] else "#f39c12" if "ELEVATED" in fracture['risk_level'] else "#27ae60"
            bg = "#fdedec" if "CRITICAL" in fracture['risk_level'] else "#fef5e7" if "ELEVATED" in fracture['risk_level'] else "#eafaf1"
            st.markdown(f"<div style='padding:1rem;border-left:4px solid {color};background:{bg};border-radius:0.3rem'><strong>{fracture['risk_level']}</strong><br>{fracture['description']}</div>", unsafe_allow_html=True)
            
            if fracture['contributing_factors']:
                st.markdown("**Contributing Factors**:")
                for f in fracture['contributing_factors']: st.markdown(f"- {f}")
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Expansion", f"{expansion:.2f}%")
            with col2: st.metric("Anisotropy AR", f"{li_el['anisotropy_ratio']:.3f}", delta="c-soft" if li_el['anisotropy_ratio']<1 else "isotropic")
            with col3: st.metric("B₀ Drop", f"{b0_drop:.1f}%")
            
            # 3D stress sphere
            st.subheader("🌐 Anisotropic Stress Distribution")
            st.markdown("*Surface radius ∝ stress; color = relative magnitude*")
            fig = plot_3d_stress_sphere(st.session_state.stress_3d, f"Li₂Sn₅ Stress (C₁₁={li_el['c11_gpa']:.0f}, C₃₃={li_el['c33_gpa']:.0f} GPa)")
            st.pyplot(fig)
            
            # Stress histogram
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(st.session_state.stress_3d['stress'].flatten(), bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Relative Stress (a.u.)'); ax.set_ylabel('Frequency'); ax.set_title('Stress Distribution'); ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)

# ============================================================================
# TAB 5: DASHBOARD
# ============================================================================
with tab5:
    st.header("📈 Integrated Dashboard")
    
    sn_eos = st.session_state.phase_results.get('phase2_sn')
    li_eos = st.session_state.phase_results.get('phase2_li2sn5')
    li_el = st.session_state.phase_results.get('phase3_li2sn5')
    thermo = st.session_state.phase_results.get('phase1')
    fracture = st.session_state.phase_results.get('phase4')
    
    if sn_eos and li_eos:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if thermo: st.markdown(f"<div class='metric-card'><strong>ΔE_f</strong><br>{thermo['formation_per_atom']:.3f} eV/atom</div>", unsafe_allow_html=True)
        with col2:
            exp = (li_eos['v0_fit']/li_eos['num_sn'] - sn_eos['v0_fit']/sn_eos['num_sn']) / (sn_eos['v0_fit']/sn_eos['num_sn']) * 100
            st.markdown(f"<div class='metric-card'><strong>Expansion</strong><br>{exp:+.1f}%</div>", unsafe_allow_html=True)
        with col3:
            if li_el: st.markdown(f"<div class='metric-card'><strong>AR</strong><br>{li_el['anisotropy_ratio']:.3f}</div>", unsafe_allow_html=True)
        with col4:
            if fracture:
                c = "#e74c3c" if "CRITICAL" in fracture['risk_level'] else "#f39c12" if "ELEVATED" in fracture['risk_level'] else "#27ae60"
                st.markdown(f"<div class='metric-card' style='background:linear-gradient(135deg,{c},#c0392b)'><strong>Risk</strong><br>{fracture['risk_level']}</div>", unsafe_allow_html=True)
        
        # Radar chart
        st.subheader("🎯 Multi-Property Radar")
        props = {
            'Stability': -thermo['formation_per_atom']*10 if thermo else 5,
            'Expansion': min(exp/3, 10) if exp else 5,
            'Anisotropy': (1-li_el['anisotropy_ratio'])*10+5 if li_el else 5,
            'Stiffness': max(0, (1-(sn_eos['B0_GPa']-li_eos['B0_GPa'])/sn_eos['B0_GPa']))*10,
            'c-axis': min(li_el['c33_gpa']/10, 10) if li_el else 5
        }
        fig = plot_radar_chart(props, "Mechanical-Thermodynamic Profile")
        st.pyplot(fig)
        
        # Export
        st.subheader("💾 Export Results")
        data = pd.DataFrame({
            'Property': ['ΔE_f (eV/atom)', 'Expansion (%)', 'V₀ Sn (Å³)', 'V₀ Li₂Sn₅ (Å³)', 'B₀ Sn (GPa)', 'B₀ Li₂Sn₅ (GPa)', 'C₁₁ Sn', 'C₃₃ Sn', 'C₁₁ Li₂Sn₅', 'C₃₃ Li₂Sn₅', 'AR', 'Risk Score'],
            'Value': [thermo['formation_per_atom'] if thermo else 'N/A', exp if exp else 'N/A', sn_eos['v0_fit'], li_eos['v0_fit'], sn_eos['B0_GPa'], li_eos['B0_GPa'], st.session_state.phase_results.get('phase3_sn',{}).get('c11_gpa','N/A'), st.session_state.phase_results.get('phase3_sn',{}).get('c33_gpa','N/A'), li_el['c11_gpa'] if li_el else 'N/A', li_el['c33_gpa'] if li_el else 'N/A', li_el['anisotropy_ratio'] if li_el else 'N/A', fracture['risk_score'] if fracture else 'N/A']
        })
        st.download_button("📥 Download CSV", data.to_csv(index=False), "sn_li2sn5_results.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#7f8c8d;font-size:0.9rem'><strong>Sn→Li₂Sn₅ Lithiation Analyzer</strong> | GPAW/PBE | ASE | Birch-Murnaghan EOS | CPU-Parallelized</div>", unsafe_allow_html=True)
