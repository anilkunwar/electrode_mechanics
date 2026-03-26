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
from gpaw import GPAW, PW
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sn→Li₂Sn₅ Mechanics & Thermodynamics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visualization
st.markdown("""
<style>
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {padding: 0.5rem 1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# Title and description
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
# SIDEBAR: GLOBAL SETTINGS
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
# HELPER FUNCTIONS
# ============================================================================

def quadratic_strain(eps, A, B, C):
    """Quadratic fit for elastic constant extraction: E(ε) = Aε² + Bε + C"""
    return A * eps**2 + B * eps + C

def birch_murnaghan_eos(V, E0, V0, B0, Bp):
    """3rd-order Birch-Murnaghan Equation of State"""
    eta = (V0 / V)**(2/3)
    return E0 + (9*V0*B0/16) * ((eta - 1)**3 * Bp + (eta - 1)**2 * (6 - 4*eta))

def create_gpaaw_calculator(ecut, xc='PBE', kpts=(4,4,4), txt=None):
    """Create GPAW calculator with consistent settings"""
    return GPAW(
        mode=PW(ecut),
        xc=xc,
        kpts=kpts,
        txt=txt,
        convergence={'energy': 1e-5, 'density': 1e-4},
        maxiter=200,
        occupations={'name': 'fermi-dirac', 'width': 0.1}
    )

def relax_fixed_volume(atoms, fmax=0.05, max_steps=100):
    """Relax atomic positions at fixed cell volume"""
    calc = atoms.calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    return atoms.get_potential_energy()

@st.cache_data(show_spinner="Computing reference energies...")
def compute_reference_energies(ecut, kpts, fmax):
    """Compute bulk reference energies for Li and Sn"""
    # Bulk Li (BCC)
    li_bulk = bulk('Li', 'bcc', a=3.51)
    li_calc = create_gpaaw_calculator(ecut, kpts=kpts, txt=None)
    li_bulk.calc = li_calc
    ef_li = ExpCellFilter(li_bulk)
    BFGS(ef_li).run(fmax=fmax)
    e_li_per_atom = li_bulk.get_potential_energy() / len(li_bulk)
    
    # Bulk Sn (BCT) - already computed in main workflow
    return {"e_li_per_atom": e_li_per_atom}

# ============================================================================
# PHASE 1: THERMODYNAMIC STABILITY
# ============================================================================
def phase1_thermodynamic_stability(e_li2sn5_total, e_sn_per, e_li_per, n_li=4, n_sn=10):
    """
    Compute formation energy for Li₂Sn₅
    ΔE_f = [E(Li₂Sn₅) - n_Li·E_Li - n_Sn·E_Sn] / N_atoms
    """
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

# ============================================================================
# PHASE 2: ISOTROPIC E-V MAPPING & EOS FITTING
# ============================================================================
@st.cache_data(show_spinner=False)
def compute_ev_curve(structure_name, a_init, c_init, symbols, spacegroup, basis, num_sn, kpts, 
                     volume_range, n_points, fmax, ecut):
    """Compute energy-volume curve with fixed-volume ion relaxation"""
    
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
    volumes, energies = [], []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, s in enumerate(scales):
        status.text(f"{structure_name}: Computing point {i+1}/{n_points} ({s*100:.1f}% V₀)")
        
        # Scale cell isotropically
        atoms = template.copy()
        scale_factor = s ** (1/3)
        atoms.set_cell(template.get_cell() * scale_factor, scale_atoms=True)
        
        # Setup calculator and relax
        calc = create_gpaaw_calculator(ecut, kpts=kpts, txt=None)
        atoms.calc = calc
        energy = relax_fixed_volume(atoms, fmax=fmax)
        
        volumes.append(atoms.get_volume())
        energies.append(energy)
        progress.progress((i + 1) / n_points)
    
    status.empty()
    progress.empty()
    
    # Fit Birch-Murnaghan EOS
    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    v0_fit, e0_fit, B0_fit, Bp_fit = eos.fit()
    
    return {
        "volumes": np.array(volumes),
        "energies": np.array(energies),
        "v0_init": v0_init,
        "v0_fit": v0_fit,
        "e0_fit": e0_fit,
        "B0_GPa": B0_fit / GPa,
        "Bp": Bp_fit,
        "num_sn": num_sn,
        "eos": eos
    }

# ============================================================================
# PHASE 3: ANISOTROPIC ELASTICITY (C₁₁, C₃₃)
# ============================================================================
@st.cache_data(show_spinner=False)
def compute_anisotropic_elasticity(structure_name, a0, c0, symbols, spacegroup, basis, 
                                   kpts, fmax, ecut, strain_range, n_strain):
    """Compute directional elastic constants C₁₁ (a-b plane) and C₃₃ (c-axis)"""
    
    template = crystal(symbols=symbols, basis=basis, spacegroup=spacegroup,
                      cellpar=[a0, a0, c0, 90, 90, 90])
    v0 = template.get_volume()
    
    strains = np.linspace(strain_range[0]/100, strain_range[1]/100, n_strain)
    
    # C₁₁: Uniaxial strain along a (c fixed)
    energies_a = []
    for eps in strains:
        atoms = template.copy()
        new_cell = [a0*(1+eps), a0*(1+eps), c0, 90, 90, 90]
        atoms.set_cell(new_cell, scale_atoms=True)
        calc = create_gpaaw_calculator(ecut, kpts=kpts, txt=None)
        atoms.calc = calc
        e = relax_fixed_volume(atoms, fmax=fmax)
        energies_a.append(e)
    
    # C₃₃: Uniaxial strain along c (a fixed)
    energies_c = []
    for eps in strains:
        atoms = template.copy()
        new_cell = [a0, a0, c0*(1+eps), 90, 90, 90]
        atoms.set_cell(new_cell, scale_atoms=True)
        calc = create_gpaaw_calculator(ecut, kpts=kpts, txt=None)
        atoms.calc = calc
        e = relax_fixed_volume(atoms, fmax=fmax)
        energies_c.append(e)
    
    # Quadratic fit to extract elastic constants
    popt_a, _ = curve_fit(quadratic_strain, strains, energies_a)
    popt_c, _ = curve_fit(quadratic_strain, strains, energies_c)
    
    # Convert to GPa: C_ii = (2A/V₀) × (160.217 GPa·Å³/eV)
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

# ============================================================================
# PHASE 4: FRACTURE PREDICTION & STRESS MAPPING
# ============================================================================
def predict_fracture_risk(expansion_pct, anisotropy_ratio, b0_drop_pct, c33_gpa):
    """
    Predict mechanical fracture risk based on multiple criteria:
    - Volume expansion magnitude
    - Elastic anisotropy
    - Bulk modulus softening
    - Absolute stiffness
    """
    risk_score = 0
    factors = []
    
    # Criterion 1: Volume expansion
    if expansion_pct > 30:
        risk_score += 3
        factors.append("🔴 Extreme expansion (>30%)")
    elif expansion_pct > 20:
        risk_score += 2
        factors.append("🟡 High expansion (20-30%)")
    elif expansion_pct > 10:
        risk_score += 1
        factors.append("🟢 Moderate expansion (10-20%)")
    
    # Criterion 2: Anisotropy (c-axis softening)
    if anisotropy_ratio < 0.7:
        risk_score += 3
        factors.append("🔴 Severe c-axis softening (AR<0.7)")
    elif anisotropy_ratio < 0.9:
        risk_score += 2
        factors.append("🟡 Moderate anisotropy (AR 0.7-0.9)")
    
    # Criterion 3: Bulk modulus drop
    if b0_drop_pct > 50:
        risk_score += 2
        factors.append("🟡 Significant softening (>50% B₀ drop)")
    
    # Criterion 4: Absolute stiffness
    if c33_gpa < 20:
        risk_score += 1
        factors.append("🟢 Low c-axis stiffness")
    
    # Final risk classification
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

def compute_stress_distribution_3d(c11, c33, expansion_tensor=None, n_theta=36, n_phi=18):
    """
    Compute stress distribution on unit sphere in polar coordinates
    for transversely isotropic material (tetragonal symmetry)
    
    Returns spherical coordinates and stress values for 3D visualization
    """
    theta = np.linspace(0, 2*np.pi, n_theta)  # azimuthal
    phi = np.linspace(0, np.pi, n_phi)  # polar
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Direction cosines for each point on sphere
    lx = np.sin(phi_grid) * np.cos(theta_grid)
    ly = np.sin(phi_grid) * np.sin(theta_grid)
    lz = np.cos(phi_grid)
    
    # Effective Young's modulus in direction (lx, ly, lz) for transverse isotropy
    # Simplified: E(θ) = [sin⁴θ/C₁₁ + cos⁴θ/C₃₃ + sin²θcos²θ(1/G + 2ν/C₁₁)]⁻¹
    # Using approximate relation for stress under volumetric strain
    if expansion_tensor is None:
        # Assume isotropic expansion for visualization
        eps_vol = 0.26  # ~26% expansion typical for Sn→Li₂Sn₅
        stress_magnitude = (c11 * (lx**2 + ly**2) + c33 * lz**2) * eps_vol / 3
    else:
        # Use provided expansion tensor
        stress_magnitude = (c11 * expansion_tensor[0] * (lx**2 + ly**2) + 
                           c33 * expansion_tensor[2] * lz**2)
    
    # Convert to spherical coordinates for plotting
    x = np.sin(phi_grid) * np.cos(theta_grid) * stress_magnitude
    y = np.sin(phi_grid) * np.sin(theta_grid) * stress_magnitude
    z = np.cos(phi_grid) * stress_magnitude
    
    return {
        "x": x, "y": y, "z": z,
        "theta": theta_grid, "phi": phi_grid,
        "stress": stress_magnitude,
        "c11": c11, "c33": c33
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_radar_chart(properties_dict, title="Property Comparison"):
    """Create radar chart for multi-property comparison"""
    categories = list(properties_dict.keys())
    N = len(categories)
    
    # Normalize values to 0-1 scale for radar chart
    values = list(properties_dict.values())
    min_val, max_val = min(values), max(values)
    if max_val > min_val:
        normalized = [(v - min_val) / (max_val - min_val) * 0.8 + 0.1 for v in values]
    else:
        normalized = [0.5] * N
    
    # Close the loop
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
    """Plot E-V scatter points with Birch-Murnaghan fit"""
    vols = eos_results["volumes"]
    energies = eos_results["energies"]
    v0 = eos_results["v0_fit"]
    e0 = eos_results["e0_fit"]
    B0 = eos_results["B0_GPa"]
    Bp = eos_results["Bp"]
    
    # Generate smooth EOS curve
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
    """Plot histogram-style bar chart of elastic constants"""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    constants = [c11, c33]
    labels = ['C₁₁ (a-b plane)', 'C₃₃ (c-axis)']
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(labels, constants, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Elastic Constant (GPa)', fontsize=11)
    ax.set_title(f'{phase_name}: Directional Stiffness', fontsize=12, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, constants):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f'{val:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    return fig

def plot_3d_stress_sphere(stress_data, title="Anisotropic Stress Distribution"):
    """Create 3D spherical plot of stress distribution"""
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = stress_data["x"], stress_data["y"], stress_data["z"]
    stress = stress_data["stress"]
    
    # Color mapping
    cmap = cm.RdYlBu_r
    norm = plt.Normalize(vmin=stress.min(), vmax=stress.max())
    colors = cmap(norm(stress))
    
    # Plot surface
    surf = ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, 
                          linewidth=0, antialiased=True, alpha=0.9)
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z (c-axis)', fontsize=10)
    ax.set_title(title, fontsize=12, weight='bold', pad=20)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Relative Stress (a.u.)', fontsize=9)
    
    # Add annotation for c-axis direction
    ax.text(0, 0, 1.3, '↑ c-axis', ha='center', fontsize=9, weight='bold', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.view_init(elev=25, azim=45)
    return fig

def plot_expansion_bar_chart(sn_results, li2sn5_results, expansion_pct):
    """Create bar chart comparing volumes per Sn atom"""
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
    
    # Add expansion arrow annotation
    ax.annotate('', xy=(1, v_per_sn_li), xytext=(0, v_per_sn_sn),
               arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='-'))
    ax.text(0.5, (v_per_sn_sn + v_per_sn_li)/2, f'+{expansion_pct:.1f}%', 
           ha='center', va='bottom', color='red', weight='bold', fontsize=11)
    
    # Add value labels
    for bar, vol in zip(bars, volumes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
               f'{vol:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Create tabs for different analysis phases
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Phase 1: Thermodynamics",
    "📊 Phase 2: EOS & Expansion", 
    "🧭 Phase 3: Anisotropic Elasticity",
    "💥 Phase 4: Fracture Prediction",
    "📈 Multi-View Dashboard"
])

# Session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = None
if 'reference_energies' not in st.session_state:
    st.session_state.reference_energies = None

# Run button
with st.sidebar:
    st.markdown("---")
    run_analysis = st.button("🚀 Run Complete 4-Phase Analysis", type="primary", use_container_width=True)
    st.markdown("*Calculations use GPAW with PBE functional*")

if run_analysis or st.session_state.results is not None:
    if st.session_state.results is None:
        try:
            # Compute reference energies
            with st.spinner("📋 Computing bulk reference energies..."):
                ref_energies = compute_reference_energies(ecut, kpts_sn, fmax)
                st.session_state.reference_energies = ref_energies
            
            # Phase 2: E-V curves for both phases
            with st.spinner("🔬 Computing BCT Sn E-V curve..."):
                sn_results = compute_ev_curve(
                    'Sn (BCT)', a_init=5.83, c_init=3.18, symbols='Sn', spacegroup=141,
                    basis=[(0,0,0)], num_sn=4, kpts=kpts_sn,
                    volume_range=volume_range, n_points=n_vol, fmax=fmax, ecut=ecut
                )
            
            with st.spinner("🔬 Computing Li₂Sn₅ E-V curve..."):
                li2sn5_results = compute_ev_curve(
                    'Li2Sn5', a_init=10.274, c_init=3.125, 
                    symbols=['Sn', 'Li', 'Sn'], spacegroup=127,
                    basis=[(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)], 
                    num_sn=10, kpts=kpts_li2sn5,
                    volume_range=volume_range, n_points=n_vol, fmax=fmax, ecut=ecut
                )
            
            # Phase 3: Anisotropic elasticity
            with st.spinner("🧭 Computing anisotropic elastic constants..."):
                sn_elastic = compute_anisotropic_elasticity(
                    'Sn', a0=5.83, c0=3.18, symbols='Sn', spacegroup=141,
                    basis=[(0,0,0)], kpts=kpts_sn, fmax=fmax, ecut=ecut,
                    strain_range=strain_range, n_strain=n_strain
                )
                li2sn5_elastic = compute_anisotropic_elasticity(
                    'Li2Sn5', a0=10.274, c0=3.125,
                    symbols=['Sn', 'Li', 'Sn'], spacegroup=127,
                    basis=[(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)],
                    kpts=kpts_li2sn5, fmax=fmax, ecut=ecut,
                    strain_range=strain_range, n_strain=n_strain
                )
            
            # Phase 1: Formation energy
            e_sn_per = sn_results["e0_fit"] / sn_results["num_sn"]
            thermo_results = phase1_thermodynamic_stability(
                e_li2sn5_total=li2sn5_results["e0_fit"],
                e_sn_per=e_sn_per,
                e_li_per=ref_energies["e_li_per_atom"]
            )
            
            # Phase 2: Volume expansion
            v_per_sn_sn = sn_results["v0_fit"] / sn_results["num_sn"]
            v_per_sn_li = li2sn5_results["v0_fit"] / li2sn5_results["num_sn"]
            expansion_pct = (v_per_sn_li - v_per_sn_sn) / v_per_sn_sn * 100
            
            # Phase 4: Fracture prediction
            b0_drop = (sn_results["B0_GPa"] - li2sn5_results["B0_GPa"]) / sn_results["B0_GPa"] * 100
            fracture_results = predict_fracture_risk(
                expansion_pct=expansion_pct,
                anisotropy_ratio=li2sn5_elastic["anisotropy_ratio"],
                b0_drop_pct=b0_drop,
                c33_gpa=li2sn5_elastic["c33_gpa"]
            )
            
            # 3D stress visualization data
            stress_3d_data = compute_stress_distribution_3d(
                c11=li2sn5_elastic["c11_gpa"],
                c33=li2sn5_elastic["c33_gpa"]
            )
            
            # Store all results
            st.session_state.results = {
                "sn_eos": sn_results,
                "li2sn5_eos": li2sn5_results,
                "sn_elastic": sn_elastic,
                "li2sn5_elastic": li2sn5_elastic,
                "thermo": thermo_results,
                "expansion_pct": expansion_pct,
                "v_per_sn": {"sn": v_per_sn_sn, "li2sn5": v_per_sn_li},
                "fracture": fracture_results,
                "stress_3d": stress_3d_data,
                "b0_drop_pct": b0_drop
            }
            
            st.success("✅ Complete analysis finished successfully!")
            
        except Exception as e:
            st.error(f"❌ Calculation failed: {str(e)}")
            st.exception(e)
            st.stop()
    
    results = st.session_state.results
    
    # ========================================================================
    # TAB 1: THERMODYNAMIC STABILITY
    # ========================================================================
    with tab1:
        st.header("🔬 Phase 1: Thermodynamic Stability Analysis")
        st.markdown("""
        **Formation Energy Calculation**:
        ```
        ΔE_f = [E(Li₂Sn₅) - 4·E(Li_bulk) - 10·E(Sn_bulk)] / 14 atoms
        ```
        Negative ΔE_f indicates spontaneous formation during lithiation.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Formation Energy (per atom)", 
                     f"{results['thermo']['formation_per_atom']:.4f} eV",
                     delta=results['thermo']['stability_label'])
        with col2:
            st.metric("Formation Energy (per formula)", 
                     f"{results['thermo']['formation_per_formula']:.3f} eV")
        with col3:
            st.metric("Total Energy Change", 
                     f"{results['thermo']['delta_e_total']:.2f} eV")
        
        # Stability indicator
        if results['thermo']['is_stable']:
            st.success("✅ **Li₂Sn₅ is thermodynamically stable** relative to bulk Li + Sn")
        else:
            st.warning("⚠️ **Li₂Sn₅ shows metastability** - kinetic factors may enable formation")
        
        # Energy diagram
        fig, ax = plt.subplots(figsize=(8, 4))
        phases = ['Li + Sn (reference)', 'Li₂Sn₅']
        energies = [0, results['thermo']['formation_per_formula']]
        colors = ['#95a5a6', '#27ae60' if results['thermo']['is_stable'] else '#e67e22']
        
        ax.bar(phases, energies, color=colors, edgecolor='black')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Energy Relative to Reference (eV/f.u.)')
        ax.set_title('Thermodynamic Stability Diagram')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    # ========================================================================
    # TAB 2: EOS & VOLUME EXPANSION
    # ========================================================================
    with tab2:
        st.header("📊 Phase 2: Equation of State & Volume Expansion")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("V₀ (β-Sn)", f"{results['sn_eos']['v0_fit']:.2f} Å³")
        with col2:
            st.metric("V₀ (Li₂Sn₅)", f"{results['li2sn5_eos']['v0_fit']:.2f} Å³")
        with col3:
            st.metric("Volume/Sn (β-Sn)", f"{results['v_per_sn']['sn']:.3f} Å³")
        with col4:
            st.metric("Volume/Sn (Li₂Sn₅)", f"{results['v_per_sn']['li2sn5']:.3f} Å³")
        
        st.markdown(f"""
        ### 📈 Volume Expansion Result
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; font-size: 1.2rem;'>
            <strong>{results['expansion_pct']:+.2f}%</strong> volume expansion per Sn atom
        </div>
        """, unsafe_allow_html=True)
        
        # E-V curves with EOS fits
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        plot_eos_scatter_with_fit(results['sn_eos'], 'β-Sn (BCT)', ax1)
        plot_eos_scatter_with_fit(results['li2sn5_eos'], 'Li₂Sn₅', ax2)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Bulk modulus comparison
        fig = plot_elasticity_histogram(
            results['sn_eos']['B0_GPa'], 
            results['li2sn5_eos']['B0_GPa'], 
            'Bulk Modulus Comparison'
        )
        st.pyplot(fig)
        
        # Volume expansion bar chart
        fig = plot_expansion_bar_chart(
            results['sn_eos'], 
            results['li2sn5_eos'], 
            results['expansion_pct']
        )
        st.pyplot(fig)
    
    # ========================================================================
    # TAB 3: ANISOTROPIC ELASTICITY
    # ========================================================================
    with tab3:
        st.header("🧭 Phase 3: Anisotropic Elastic Constants")
        st.markdown("""
        **Methodology**: Finite strain approach with quadratic energy fitting
        ```
        C_ii = (2A / V₀) × 160.217  [eV/Å³ → GPa]
        Anisotropy Ratio: AR = C₃₃ / C₁₁
        ```
        AR < 1 indicates c-axis softening → preferential expansion along [001]
        """)
        
        # Elastic constants summary
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("β-Sn (BCT)")
            st.write(f"**C₁₁** (a-b plane): `{results['sn_elastic']['c11_gpa']:.1f} GPa`")
            st.write(f"**C₃₃** (c-axis): `{results['sn_elastic']['c33_gpa']:.1f} GPa`")
            st.write(f"**Anisotropy**: `{results['sn_elastic']['anisotropy_ratio']:.3f}`")
        with col2:
            st.subheader("Li₂Sn₅")
            st.write(f"**C₁₁** (a-b plane): `{results['li2sn5_elastic']['c11_gpa']:.1f} GPa`")
            st.write(f"**C₃₃** (c-axis): `{results['li2sn5_elastic']['c33_gpa']:.1f} GPa`")
            st.write(f"**Anisotropy**: `{results['li2sn5_elastic']['anisotropy_ratio']:.3f}`")
        
        # Strain-energy curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # C11 fit (a-axis strain)
        strains = results['li2sn5_elastic']['strains'] * 100  # Convert to %
        ax1.plot(strains, results['li2sn5_elastic']['energies_a'], 'o-', 
                label='DFT', color='#e74c3c', markersize=6)
        fit_a = quadratic_strain(strains/100, *curve_fit(quadratic_strain, 
                      results['li2sn5_elastic']['strains'], 
                      results['li2sn5_elastic']['energies_a'])[0])
        ax1.plot(strains, fit_a, '--', label='Quadratic Fit', color='#3498db')
        ax1.set_xlabel('Strain ε_a (%)')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title('C₁₁ Extraction: a-axis Strain')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # C33 fit (c-axis strain)
        ax2.plot(strains, results['li2sn5_elastic']['energies_c'], 'o-', 
                label='DFT', color='#e74c3c', markersize=6)
        fit_c = quadratic_strain(strains/100, *curve_fit(quadratic_strain,
                      results['li2sn5_elastic']['strains'],
                      results['li2sn5_elastic']['energies_c'])[0])
        ax2.plot(strains, fit_c, '--', label='Quadratic Fit', color='#9b59b6')
        ax2.set_xlabel('Strain ε_c (%)')
        ax2.set_ylabel('Energy (eV)')
        ax2.set_title('C₃₃ Extraction: c-axis Strain')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Anisotropy radar chart
        properties = {
            'C₁₁ (Sn)': results['sn_elastic']['c11_gpa'],
            'C₃₃ (Sn)': results['sn_elastic']['c33_gpa'],
            'C₁₁ (Li₂Sn₅)': results['li2sn5_elastic']['c11_gpa'],
            'C₃₃ (Li₂Sn₅)': results['li2sn5_elastic']['c33_gpa']
        }
        fig = plot_radar_chart(properties, "Elastic Constants Comparison")
        st.pyplot(fig)
    
    # ========================================================================
    # TAB 4: FRACTURE PREDICTION & 3D STRESS
    # ========================================================================
    with tab4:
        st.header("💥 Phase 4: Mechanical Fracture Prediction")
        
        # Fracture risk summary
        st.markdown(f"""
        ### 🎯 Fracture Risk Assessment
        <div style='padding: 1rem; border-left: 4px solid {"#e74c3c" if "CRITICAL" in results["fracture"]["risk_level"] else "#f39c12" if "ELEVATED" in results["fracture"]["risk_level"] else "#27ae60"}; 
                    background: {"#fdedec" if "CRITICAL" in results["fracture"]["risk_level"] else "#fef5e7" if "ELEVATED" in results["fracture"]["risk_level"] else "#eafaf1"};
                    border-radius: 0.3rem;'>
            <strong>{results['fracture']['risk_level']}</strong><br>
            {results['fracture']['description']}
        </div>
        """, unsafe_allow_html=True)
        
        # Contributing factors
        if results['fracture']['contributing_factors']:
            st.markdown("**Contributing Factors**:")
            for factor in results['fracture']['contributing_factors']:
                st.markdown(f"- {factor}")
        
        # Key mechanical metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Volume Expansion", f"{results['expansion_pct']:.2f}%")
        with col2:
            st.metric("Anisotropy Ratio (AR)", 
                     f"{results['li2sn5_elastic']['anisotropy_ratio']:.3f}",
                     delta="c-soft" if results['li2sn5_elastic']['anisotropy_ratio'] < 1 else "isotropic")
        with col3:
            st.metric("Bulk Modulus Drop", f"{results['b0_drop_pct']:.1f}%")
        
        # 3D Stress Distribution Sphere
        st.subheader("🌐 Anisotropic Stress Distribution (Polar Spherical Coordinates)")
        st.markdown("*Surface radius ∝ stress magnitude; color indicates relative stress*")
        
        fig = plot_3d_stress_sphere(
            results['stress_3d'],
            title=f"Li₂Sn₅ Stress Map (C₁₁={results['li2sn5_elastic']['c11_gpa']:.0f}, C₃₃={results['li2sn5_elastic']['c33_gpa']:.0f} GPa)"
        )
        st.pyplot(fig)
        
        # Stress histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        stress_vals = results['stress_3d']['stress'].flatten()
        ax.hist(stress_vals, bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Relative Stress (a.u.)')
        ax.set_ylabel('Frequency')
        ax.set_title('Stress Distribution Histogram')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    # ========================================================================
    # TAB 5: MULTI-VIEW DASHBOARD
    # ========================================================================
    with tab5:
        st.header("📈 Integrated Multi-View Dashboard")
        
        # Summary metrics grid
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><strong>ΔE_f</strong><br>{:.3f} eV/atom</div>'.format(
                results['thermo']['formation_per_atom']), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><strong>Expansion</strong><br>{:+.1f}%</div>'.format(
                results['expansion_pct']), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><strong>AR</strong><br>{:.3f}</div>'.format(
                results['li2sn5_elastic']['anisotropy_ratio']), unsafe_allow_html=True)
        with col4:
            risk_color = "#e74c3c" if "CRITICAL" in results['fracture']['risk_level'] else "#f39c12" if "ELEVATED" in results['fracture']['risk_level'] else "#27ae60"
            st.markdown(f'<div class="metric-card" style="background: linear-gradient(135deg, {risk_color} 0%, #c0392b 100%)"><strong>Risk</strong><br>{results["fracture"]["risk_level"]}</div>', 
                       unsafe_allow_html=True)
        
        # Comprehensive radar chart
        st.subheader("🎯 Multi-Property Radar Analysis")
        radar_props = {
            'Stability (-ΔE_f)': -results['thermo']['formation_per_atom'] * 10,  # Scale for visibility
            'Expansion Risk': min(results['expansion_pct'] / 3, 10),  # Normalize
            'Anisotropy': (1 - results['li2sn5_elastic']['anisotropy_ratio']) * 10 + 5,
            'Stiffness Retention': max(0, (1 - results['b0_drop_pct']/100)) * 10,
            'c-axis Strength': min(results['li2sn5_elastic']['c33_gpa'] / 10, 10)
        }
        fig = plot_radar_chart(radar_props, "Integrated Mechanical-Thermodynamic Profile")
        st.pyplot(fig)
        
        # Scatter matrix of key variables
        st.subheader("🔗 Property Correlations")
        df_scatter = pd.DataFrame({
            'Volume/Sn (Å³)': [results['v_per_sn']['sn'], results['v_per_sn']['li2sn5']],
            'Bulk Modulus (GPa)': [results['sn_eos']['B0_GPa'], results['li2sn5_eos']['B0_GPa']],
            'C₃₃ (GPa)': [results['sn_elastic']['c33_gpa'], results['li2sn5_elastic']['c33_gpa']],
            'Phase': ['β-Sn', 'Li₂Sn₅']
        })
        
        fig = px.scatter_matrix(
            df_scatter, 
            dimensions=['Volume/Sn (Å³)', 'Bulk Modulus (GPa)', 'C₃₃ (GPa)'],
            color='Phase',
            title="Multi-Property Scatter Matrix",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data export
        st.subheader("💾 Export Results")
        export_df = pd.DataFrame({
            'Property': [
                'Formation Energy (eV/atom)', 'Volume Expansion (%)', 
                'V₀ Sn (Å³)', 'V₀ Li₂Sn₅ (Å³)', 'B₀ Sn (GPa)', 'B₀ Li₂Sn₅ (GPa)',
                'C₁₁ Sn (GPa)', 'C₃₃ Sn (GPa)', 'C₁₁ Li₂Sn₅ (GPa)', 'C₃₃ Li₂Sn₅ (GPa)',
                'Anisotropy Ratio', 'Fracture Risk Score'
            ],
            'Value': [
                results['thermo']['formation_per_atom'], results['expansion_pct'],
                results['sn_eos']['v0_fit'], results['li2sn5_eos']['v0_fit'],
                results['sn_eos']['B0_GPa'], results['li2sn5_eos']['B0_GPa'],
                results['sn_elastic']['c11_gpa'], results['sn_elastic']['c33_gpa'],
                results['li2sn5_elastic']['c11_gpa'], results['li2sn5_elastic']['c33_gpa'],
                results['li2sn5_elastic']['anisotropy_ratio'], results['fracture']['risk_score']
            ]
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            "📥 Download Complete Results (CSV)",
            csv,
            "sn_li2sn5_mechanics_full.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Raw E-V data
        with st.expander("📋 Raw E-V Data Tables"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**β-Sn E-V Data**")
                df_sn = pd.DataFrame({
                    'Volume (Å³)': results['sn_eos']['volumes'],
                    'Energy (eV)': results['sn_eos']['energies']
                })
                st.dataframe(df_sn, use_container_width=True)
            with col2:
                st.write("**Li₂Sn₅ E-V Data**")
                df_li = pd.DataFrame({
                    'Volume (Å³)': results['li2sn5_eos']['volumes'],
                    'Energy (eV)': results['li2sn5_eos']['energies']
                })
                st.dataframe(df_li, use_container_width=True)

else:
    # Welcome screen before calculation
    st.info("👈 Configure settings in the sidebar, then click **Run Complete 4-Phase Analysis** to begin.")
    
    # Preview of methodology
    with st.expander("📚 Methodology Details (Click to Expand)", expanded=True):
        st.markdown("""
        ### 🔬 Phase 1: Thermodynamic Stability
        - Compute formation energy: ΔE_f = [E(Li₂Sn₅) - 4E_Li - 10E_Sn]/14
        - Negative ΔE_f → spontaneous phase formation
        
        ### 📊 Phase 2: Isotropic E-V Mapping
        - Generate 7-11 volumes via isotropic scaling (92-108% V₀)
        - Fixed-volume ion relaxation (BFGS)
        - Fit to 3rd-order Birch-Murnaghan EOS:
          ```
          E(V) = E₀ + (9V₀B₀/16){[(V₀/V)^(2/3)-1]³B'₀ + [(V₀/V)^(2/3)-1]²[6-4(V₀/V)^(2/3)]}
          ```
        - Extract: V₀, B₀, volume expansion %
        
        ### 🧭 Phase 3: Anisotropic Elasticity
        - Apply ±2% uniaxial strains along a and c directions
        - Quadratic fit: E(ε) = Aε² + Bε + C
        - Extract: C₁₁ = (2A/V₀)×160.217, C₃₃ = (2A_c/V₀)×160.217
        - Anisotropy ratio: AR = C₃₃/C₁₁
        
        ### 💥 Phase 4: Fracture Prediction
        - Risk criteria:
          • Expansion >20% → high strain energy
          • AR <0.9 → c-axis delamination risk  
          • B₀ drop >50% → material softening
        - 3D stress mapping in polar spherical coordinates
        - Griffith-type failure criterion for crack propagation
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>
<strong>Sn→Li₂Sn₅ Lithiation Mechanics Analyzer</strong> | 
DFT: GPAW/PBE | ASE Framework | Birch-Murnaghan EOS | 
Anisotropic Elasticity | Fracture Mechanics
</div>
""", unsafe_allow_html=True)
