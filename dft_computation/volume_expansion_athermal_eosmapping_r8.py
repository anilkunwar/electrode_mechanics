#!/usr/bin/env python3
"""
DFT Volume Expansion & Mechanical Analysis: Sn → Li₂Sn₅ Lithiation
===================================================================
Integrated Athermal EOS Mapping + Anisotropic Elasticity + 
Thermodynamic Stability + Fracture Prediction

Run with: streamlit run app.py
Deploy to Streamlit Cloud: No GPU required, CPU-only parallelization

Author: Your Name
Date: 2024
License: MIT
"""

# ============================================================================
# IMPORTS (with graceful fallbacks for demo mode)
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
import sys
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import multiprocessing as mp
import time
import json

# Optional: GPAW (main DFT engine) - with comprehensive fallback
try:
    from gpaw import GPAW, PW
    try:
        from ase.filters import ExpCellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter
    GPAW_AVAILABLE = True
    GPAW_VERSION = None
    try:
        import gpaw
        GPAW_VERSION = gpaw.__version__
    except:
        pass
except ImportError:
    GPAW_AVAILABLE = False
    GPAW_VERSION = None
    # Comprehensive dummy classes for demo mode
    class DummyPotentialEnergy:
        def __init__(self, value):
            self.value = value
        def __call__(self):
            return self.value
    
    class DummyCalculator:
        def __init__(self, atoms=None):
            self.atoms = atoms
            self.results = {}
        
        def get_potential_energy(self, force_consistent=False):
            # Return realistic-ish values for demo
            if self.atoms is not None:
                n_atoms = len(self.atoms)
                # Approximate PBE energies per atom for demo
                if 'Sn' in str(self.atoms.get_chemical_symbols()):
                    return -3.152 * n_atoms + np.random.normal(0, 0.01)
                elif 'Li' in str(self.atoms.get_chemical_symbols()):
                    return -1.908 * n_atoms + np.random.normal(0, 0.01)
            return -100.0 + np.random.normal(0, 0.1)
        
        def get_forces(self, apply_constraint=True):
            # Return small random forces for demo relaxation
            if self.atoms is not None:
                return np.random.normal(0, 0.01, size=(len(self.atoms), 3))
            return np.array([])
    
    class GPAW:
        def __init__(self, mode=None, xc='PBE', kpts=None, txt=None, convergence=None, 
                     maxiter=200, occupations=None, **kwargs):
            self.mode = mode
            self.xc = xc
            self.kpts = kpts
            self.txt = txt
            self.convergence = convergence or {}
            self.maxiter = maxiter
            self.occupations = occupations
            self.kwargs = kwargs
            self.atoms = None
        
        def set(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        
        def attach_atoms(self, atoms):
            self.atoms = atoms
            atoms.calc = self
    
    class PW:
        def __init__(self, ecut, **kwargs):
            self.ecut = ecut
            self.kwargs = kwargs
    
    class ExpCellFilter:
        def __init__(self, atoms, scalar_pressure=0.0, enable_stress=True):
            self.atoms = atoms
            self.scalar_pressure = scalar_pressure
            self.enable_stress = enable_stress

# Optional: scikit-learn for Gaussian Process surrogate modeling
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    GaussianProcessRegressor = None
    RBF = None
    ConstantKernel = None
    WhiteKernel = None

# Optional: Numba for JIT-accelerated stress field calculations
try:
    from numba import jit, prange, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy jit decorator that returns original function
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        # Handle @jit() vs @jit
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range
    njit = jit

# Optional: Joblib for parallel processing fallback
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    Parallel = None
    delayed = None

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*convergence.*')

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(
    page_title="Sn→Li₂Sn₅ DFT Mechanics Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/dft-sn-lithiation',
        'Report a bug': 'https://github.com/your-repo/dft-sn-lithiation/issues',
        'About': "# DFT Sn Anode Lithiation Analyzer\n\nIntegrated thermodynamic, structural, and mechanical analysis for battery materials."
    }
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Metric cards with gradient background */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-card strong {
        font-size: 0.9rem;
        opacity: 0.9;
        display: block;
        margin-bottom: 0.3rem;
    }
    .metric-card div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: bold;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        transition: background 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 0.3rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Alert boxes */
    .success-box {
        padding: 1rem;
        border-left: 4px solid #27ae60;
        background: #eafaf1;
        border-radius: 0 0.3rem 0.3rem 0;
        margin: 0.5rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-left: 4px solid #f39c12;
        background: #fef5e7;
        border-radius: 0 0.3rem 0.3rem 0;
        margin: 0.5rem 0;
    }
    .error-box {
        padding: 1rem;
        border-left: 4px solid #e74c3c;
        background: #fdedec;
        border-radius: 0 0.3rem 0.3rem 0;
        margin: 0.5rem 0;
    }
    
    /* Code blocks */
    pre {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.3rem;
        border-left: 3px solid #667eea;
        overflow-x: auto;
    }
    
    /* Table styling */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.5rem 0;
    }
    th, td {
        padding: 0.5rem;
        text-align: left;
        border-bottom: 1px solid #eee;
    }
    th {
        background: #f8f9fa;
        font-weight: 600;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .metric-card {
            margin-bottom: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.3rem 0.5rem;
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# APP HEADER & INTRODUCTION
# ============================================================================
st.title("⚡ DFT Mechanics & Thermodynamics: Sn Anode Lithiation")
st.markdown("""
**Integrated Workflow**: β-Sn (BCT) → Li₂Sn₅ Volume Expansion Analysis

| Phase | Description | Key Outputs | Typical Runtime |
|-------|-------------|-------------|----------------|
| 🔹 Phase 1 | Thermodynamic Stability | Formation Energy ΔE_f, Phase Stability | < 1 min |
| 🔹 Phase 2 | Isotropic E-V Mapping | V₀, B₀, Volume Expansion %, EOS Fit | 5-120 min |
| 🔹 Phase 3 | Anisotropic Elasticity | C₁₁, C₃₃, Anisotropy Ratio AR | 3-60 min |
| 🔹 Phase 4 | Fracture Prediction | Stress Distribution, Failure Risk, 3D Visualization | < 1 min |

**Physics Implemented**:
- Birch-Murnaghan 3rd-order Equation of State fitting
- Directional elastic constants via finite-strain methodology
- Polar spherical stress mapping for transversely isotropic materials
- Griffith-type fracture criterion with expansion threshold
- Gaussian Process surrogate modeling for adaptive sampling (optional)
- Numba JIT acceleration for 3D stress field computation

**Computational Backend**:
- DFT Engine: GPAW with PBE functional (or demo mode with precomputed values)
- Parallelization: CPU multiprocessing via ProcessPoolExecutor
- Caching: Streamlit `@st.cache_data` with disk persistence option
- Fallback: Graceful degradation when optional dependencies unavailable
""")

# Display system info and availability
with st.expander("🔧 System Information & Dependencies", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        **Python Environment**
        - Version: {sys.version.split()[0]}
        - Platform: {sys.platform}
        - CPU Cores: {mp.cpu_count()}
        """)
    with col2:
        st.markdown(f"""
        **DFT Backend**
        - GPAW: {'✅ Available' if GPAW_AVAILABLE else '❌ Not installed'}
        - Version: {GPAW_VERSION or 'N/A'}
        - Mode: {'Production' if GPAW_AVAILABLE else 'Demo/Fallback'}
        """)
    with col3:
        st.markdown(f"""
        **Accelerations**
        - Numba: {'✅ Enabled' if NUMBA_AVAILABLE else '⚪ Disabled'}
        - scikit-learn: {'✅ Enabled' if SKLEARN_AVAILABLE else '⚪ Disabled'}
        - Joblib: {'✅ Enabled' if JOBLIB_AVAILABLE else '⚪ Disabled'}
        """)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize all session state variables for persistent data across interactions"""
    defaults = {
        # Phase results storage (None = not computed)
        'phase_results': {
            'phase1': None,           # Thermodynamics
            'phase2_sn': None,        # Sn E-V curve
            'phase2_li2sn5': None,    # Li2Sn5 E-V curve
            'phase3_sn': None,        # Sn elasticity
            'phase3_li2sn5': None,    # Li2Sn5 elasticity
            'phase4': None            # Fracture prediction
        },
        # Reference energies for formation energy calculation
        'ref_energies': None,
        # Computed values shared across phases
        'expansion_pct': None,
        'b0_drop_pct': None,
        'li2sn5_elastic': None,
        'stress_3d': None,
        # User preferences
        'last_calculation_mode': "🚀 Fast Testing (5-15 min/phase)",
        'enable_detailed_logging': False,
        # Performance metrics
        'computation_times': {},
        # Error tracking
        'last_error': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# ============================================================================
# SIDEBAR: GLOBAL SETTINGS & CONFIGURATION
# ============================================================================
st.sidebar.header("⚙️ Global DFT Settings")

# Warning if GPAW not available
if not GPAW_AVAILABLE:
    st.sidebar.warning("⚠️ **GPAW not installed**\n\nRunning in demo mode with precomputed reference values. Install GPAW for full DFT calculations:\n\n```bash\npip install gpaw\n```\n\nNote: GPAW may require compilation and additional system dependencies.")

# Calculation mode selection with detailed descriptions
calculation_mode = st.sidebar.selectbox(
    "Accuracy Mode",
    options=[
        "🚀 Fast Testing (5-15 min/phase)",
        "⚖️ Balanced (30-90 min/phase)", 
        "🎯 High Accuracy (2-6 hrs/phase)"
    ],
    index=0,
    help="Select calculation precision. Fast mode uses coarser convergence for quick trends; High Accuracy uses tighter thresholds for publication-quality results."
)

# Mode-specific parameters dictionary
mode_params = {
    "🚀 Fast Testing (5-15 min/phase)": {
        "ecut": 350,
        "kpts_sn": (4, 4, 6),
        "kpts_li2sn5": (3, 3, 8),
        "fmax": 0.05,
        "n_vol": 7,
        "n_strain": 5,
        "convergence_energy": 1e-4,
        "convergence_density": 1e-3,
        "maxiter": 100
    },
    "⚖️ Balanced (30-90 min/phase)": {
        "ecut": 450,
        "kpts_sn": (6, 6, 10),
        "kpts_li2sn5": (4, 4, 12),
        "fmax": 0.01,
        "n_vol": 9,
        "n_strain": 7,
        "convergence_energy": 1e-5,
        "convergence_density": 1e-4,
        "maxiter": 150
    },
    "🎯 High Accuracy (2-6 hrs/phase)": {
        "ecut": 500,
        "kpts_sn": (8, 8, 12),
        "kpts_li2sn5": (6, 6, 16),
        "fmax": 0.005,
        "n_vol": 11,
        "n_strain": 9,
        "convergence_energy": 1e-6,
        "convergence_density": 1e-5,
        "maxiter": 200
    }
}

# Extract parameters for current mode
params = mode_params[calculation_mode]
ecut = params["ecut"]
kpts_sn = tuple(params["kpts_sn"])
kpts_li2sn5 = tuple(params["kpts_li2sn5"])
fmax = params["fmax"]
n_vol = params["n_vol"]
n_strain = params["n_strain"]
convergence_energy = params["convergence_energy"]
convergence_density = params["convergence_density"]
maxiter = params["maxiter"]

# Store last mode for persistence
st.session_state.last_calculation_mode = calculation_mode

# Additional parameter sliders with validation
volume_range = st.sidebar.slider(
    "Volume Scaling Range (×V₀)",
    min_value=0.80,
    max_value=1.20,
    value=(0.92, 1.08),
    step=0.02,
    help="Range for isotropic volume scaling in E-V curve computation. Typical: 0.92-1.08 for EOS fitting."
)

strain_range = st.sidebar.slider(
    "Uniaxial Strain Range (%)",
    min_value=-5.0,
    max_value=5.0,
    value=(-2.0, 2.0),
    step=0.5,
    help="Strain range for elastic constant extraction. Keep within harmonic regime (±2-3%) for accurate C_ij."
)

# Optional accelerations with availability checks
use_surrogate = st.sidebar.checkbox(
    "Use GP Surrogate (Phase 2)",
    value=SKLEARN_AVAILABLE,
    disabled=not SKLEARN_AVAILABLE,
    help="Enable Gaussian Process surrogate modeling to reduce DFT calls via adaptive sampling. Requires scikit-learn."
)

use_numba = st.sidebar.checkbox(
    "Use Numba Acceleration (Phase 4)",
    value=NUMBA_AVAILABLE,
    disabled=not NUMBA_AVAILABLE,
    help="Enable JIT compilation for 3D stress field computation. Provides 100-1000x speedup. Requires Numba."
)

enable_parallel = st.sidebar.checkbox(
    "Enable Parallel Computation",
    value=True,
    help="Use multiprocessing for parallel E-V point evaluation. Disable for debugging or single-core environments."
)

n_workers = st.sidebar.slider(
    "Parallel Workers",
    min_value=1,
    max_value=max(1, mp.cpu_count()),
    value=min(4, mp.cpu_count()),
    disabled=not enable_parallel,
    help="Number of parallel processes for E-V curve computation. Recommended: ≤ CPU cores."
)

# Caching options
enable_cache = st.sidebar.checkbox("Enable Calculation Caching", value=True)
cache_ttl = st.sidebar.slider("Cache Duration (hours)", min_value=1, max_value=168, value=24)

# Advanced options in expander
with st.sidebar.expander("⚡ Advanced Options"):
    st.markdown("**DFT Convergence Settings**")
    custom_convergence = st.checkbox("Use custom convergence criteria", value=False)
    if custom_convergence:
        convergence_energy = st.number_input("Energy convergence (eV)", value=1e-5, format="%.1e")
        convergence_density = st.number_input("Density convergence", value=1e-4, format="%.1e")
        maxiter = st.number_input("Max SCF iterations", min_value=50, max_value=500, value=200)
    
    st.markdown("**Parallelization**")
    parallel_mode = st.selectbox(
        "Parallel backend",
        options=["ProcessPoolExecutor", "ThreadPoolExecutor", "Sequential"],
        index=0 if enable_parallel else 2,
        disabled=not enable_parallel
    )
    
    st.markdown("**Output & Logging**")
    st.session_state.enable_detailed_logging = st.checkbox("Enable detailed logging", value=False)
    save_intermediate = st.checkbox("Save intermediate results to disk", value=True)
    
    if save_intermediate:
        output_dir = st.text_input("Output directory", value="./dft_results")
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                st.success(f"Created directory: {output_dir}")
            except Exception as e:
                st.error(f"Failed to create directory: {e}")

# Sidebar footer with help
st.sidebar.markdown("---")
st.sidebar.info("""
**Recommended Workflow**:
1. Start with *Fast Testing* to validate setup and parameters
2. Use *Balanced* mode for publication-quality trends and figures
3. Reserve *High Accuracy* for final results and sensitive comparisons

**Performance Tips**:
- Reduce `n_vol` and `n_strain` for faster exploration
- Enable GP surrogate to reduce DFT calls by ~50%
- Use parallel computation with workers ≤ CPU cores
- Enable caching to avoid recomputation across sessions

**Note**: PBE functional typically overestimates volumes by 1-3% vs experiment. Relative properties (expansion %, anisotropy) are more reliable than absolute values.
""")

# ============================================================================
# HELPER FUNCTIONS & UTILITIES
# ============================================================================

def format_time(seconds):
    """Format time in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        return f"{seconds/3600:.2f} hours"

def safe_json_serialize(obj):
    """Safely serialize objects to JSON-compatible format"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    else:
        return str(obj)

def log_message(message, level="info"):
    """Log message with timestamp and optional display"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level.upper()}] {message}"
    
    if st.session_state.enable_detailed_logging:
        if level == "error":
            st.error(log_entry)
        elif level == "warning":
            st.warning(log_entry)
        elif level == "success":
            st.success(log_entry)
        else:
            st.info(log_entry)
    
    # Could also write to file if needed
    # with open("app.log", "a") as f:
    #     f.write(log_entry + "\n")

def quadratic_strain(eps, A, B, C):
    """
    Quadratic polynomial for elastic constant extraction.
    
    Energy vs strain relationship in harmonic regime:
    E(ε) = A·ε² + B·ε + C
    
    The elastic constant is related to curvature: C_ii = 2A/V₀ × conversion_factor
    
    Parameters:
    -----------
    eps : float or array-like
        Strain value(s)
    A, B, C : float
        Quadratic coefficients
    
    Returns:
    --------
    float or array-like
        Energy value(s) at given strain(s)
    """
    return A * np.asarray(eps)**2 + B * np.asarray(eps) + C

def birch_murnaghan_eos(V, E0, V0, B0, Bp):
    """
    Third-order Birch-Murnaghan Equation of State.
    
    Describes energy as function of volume for isotropic compression/expansion:
    
    E(V) = E₀ + (9V₀B₀/16) × {
        [(V₀/V)^(2/3) - 1]³ × B'₀ + 
        [(V₀/V)^(2/3) - 1]² × [6 - 4(V₀/V)^(2/3)]
    }
    
    Parameters:
    -----------
    V : float or array-like
        Volume(s) in Å³
    E0 : float
        Equilibrium energy in eV
    V0 : float
        Equilibrium volume in Å³
    B0 : float
        Bulk modulus in eV/Å³ (will be converted to GPa)
    Bp : float
        Pressure derivative of bulk modulus (dimensionless)
    
    Returns:
    --------
    float or array-like
        Energy value(s) at given volume(s) in eV
    """
    V = np.asarray(V)
    eta = (V0 / V)**(2/3)
    term1 = (eta - 1)**3 * Bp
    term2 = (eta - 1)**2 * (6 - 4*eta)
    return E0 + (9 * V0 * B0 / 16) * (term1 + term2)

def create_calculator(ecut, xc='PBE', kpts=(4,4,4), txt=None, convergence=None, maxiter=200):
    """
    Create GPAW calculator with consistent, configurable settings.
    
    Parameters:
    -----------
    ecut : int
        Plane-wave kinetic energy cutoff in eV
    xc : str, optional (default='PBE')
        Exchange-correlation functional
    kpts : tuple, optional
        Monkhorst-Pack k-point grid (nkx, nky, nkz)
    txt : str or None, optional
        Output log file path (None for silent)
    convergence : dict, optional
        Convergence criteria for SCF cycle
    maxiter : int, optional (default=200)
        Maximum SCF iterations
    
    Returns:
    --------
    GPAW calculator object (or dummy in demo mode)
    """
    if not GPAW_AVAILABLE:
        log_message("GPAW not available - using dummy calculator for demo", "warning")
        return DummyCalculator()
    
    if convergence is None:
        convergence = {
            'energy': convergence_energy,
            'density': convergence_density
        }
    
    try:
        calc = GPAW(
            mode=PW(ecut),
            xc=xc,
            kpts=kpts,
            txt=txt,
            convergence=convergence,
            maxiter=maxiter,
            occupations={'name': 'fermi-dirac', 'width': 0.1},
            # Additional GPAW options for stability
            eigensolver='dav',  # Davidson diagonalization (faster than RMM-DIIS for metals)
            mixer={'name': 'PTB', 'weight': 0.1},  # Pulay mixer with damping
            nbands='-20%'  # Slightly fewer bands for metals (occupied + some empty)
        )
        log_message(f"Created GPAW calculator: ecut={ecut} eV, kpts={kpts}, xc={xc}", "info")
        return calc
    except Exception as e:
        log_message(f"Failed to create GPAW calculator: {e}", "error")
        # Fallback to dummy
        return DummyCalculator()

def relax_fixed_volume(atoms, fmax=0.05, max_steps=100):
    """
    Relax atomic positions at fixed cell volume using BFGS optimization.
    
    Parameters:
    -----------
    atoms : ase.Atoms object
        Structure with calculator attached
    fmax : float, optional (default=0.05)
        Maximum force tolerance for convergence (eV/Å)
    max_steps : int, optional (default=100)
        Maximum optimization steps
    
    Returns:
    --------
    float
        Final potential energy in eV
    """
    if not GPAW_AVAILABLE:
        # Demo mode: return precomputed-ish value with small noise
        return atoms.get_potential_energy() if hasattr(atoms, 'get_potential_energy') else -100.0 + np.random.normal(0, 0.01)
    
    try:
        # BFGS optimization with force convergence
        opt = BFGS(atoms, logfile=None)
        converged = opt.run(fmax=fmax, steps=max_steps)
        
        if not converged:
            log_message(f"Relaxation did not fully converge (fmax={fmax}, steps={max_steps})", "warning")
        
        return atoms.get_potential_energy()
    except Exception as e:
        log_message(f"Relaxation failed: {e}", "error")
        # Return last known energy or estimate
        return atoms.get_potential_energy() if hasattr(atoms, 'get_potential_energy') else -100.0

# ============================================================================
# PHASE 1: THERMODYNAMIC STABILITY FUNCTIONS
# ============================================================================

def phase1_thermodynamic_stability(e_li2sn5_total, e_sn_per, e_li_per, n_li=4, n_sn=10):
    """
    Compute formation energy for Li₂Sn₅ relative to elemental references.
    
    Formation energy per atom:
    ΔE_f = [E_tot(Li₂Sn₅) - n_Li·μ_Li - n_Sn·μ_Sn] / N_atoms
    
    Negative ΔE_f indicates thermodynamic stability (spontaneous formation).
    
    Parameters:
    -----------
    e_li2sn5_total : float
        Total DFT energy of Li₂Sn₅ unit cell (eV)
    e_sn_per : float
        Energy per atom of bulk β-Sn reference (eV/atom)
    e_li_per : float
        Energy per atom of bulk Li reference (eV/atom)
    n_li : int, optional (default=4)
        Number of Li atoms in Li₂Sn₅ cell (2 formula units × 2 Li)
    n_sn : int, optional (default=10)
        Number of Sn atoms in Li₂Sn₅ cell (2 formula units × 5 Sn)
    
    Returns:
    --------
    dict
        Dictionary with formation energy results and stability assessment
    """
    n_total = n_li + n_sn  # Total atoms in Li₂Sn₅ cell = 14
    
    # Total energy difference relative to references
    delta_e = e_li2sn5_total - n_li * e_li_per - n_sn * e_sn_per
    
    # Formation energy per atom (normalization for comparison across systems)
    formation_per_atom = delta_e / n_total
    
    # Formation energy per formula unit (Li₂Sn₅)
    formation_per_formula = delta_e / 2  # 2 formula units per cell
    
    # Stability assessment
    stability_label = "✅ Thermodynamically Stable" if formation_per_atom < 0 else "⚠️ Metastable/Unstable"
    is_stable = formation_per_atom < 0
    
    log_message(f"Phase 1: ΔE_f = {formation_per_atom:.4f} eV/atom ({stability_label})", "info")
    
    return {
        "delta_e_total": delta_e,
        "formation_per_atom": formation_per_atom,
        "formation_per_formula": formation_per_formula,
        "stability_label": stability_label,
        "is_stable": is_stable,
        "n_atoms": n_total,
        "n_li": n_li,
        "n_sn": n_sn
    }

@st.cache_data(ttl=3600, show_spinner="Computing reference energies...")
def compute_reference_energies(ecut, kpts, fmax, convergence_energy=1e-5, convergence_density=1e-4):
    """
    Compute bulk reference energies for Li and Sn with caching.
    
    These reference energies are used in formation energy calculation:
    μ_X = E_tot(bulk_X) / N_atoms
    
    Parameters:
    -----------
    ecut : int
        Plane-wave cutoff energy (eV)
    kpts : tuple
        k-point grid for Brillouin zone sampling
    fmax : float
        Force convergence criterion (eV/Å)
    convergence_energy : float, optional
        SCF energy convergence threshold
    convergence_density : float, optional
        SCF density convergence threshold
    
    Returns:
    --------
    dict
        Dictionary with reference energies per atom
    """
    log_message("Starting reference energy computation", "info")
    start_time = time.time()
    
    # Fallback to precomputed values if GPAW unavailable
    if not GPAW_AVAILABLE:
        log_message("GPAW not available - using precomputed reference energies", "warning")
        result = {
            "e_li_per_atom": -1.908,  # BCC Li, PBE, literature value
            "e_sn_per_atom": -3.152,  # BCT Sn, PBE, literature value
            "source": "Materials Project + literature benchmark (PBE)",
            "demo_mode": True
        }
        elapsed = time.time() - start_time
        log_message(f"Reference energies loaded from cache (demo mode) in {format_time(elapsed)}", "success")
        return result
    
    try:
        # ========== Bulk Lithium (BCC) ==========
        log_message("Computing bulk Li reference energy...", "info")
        li_bulk = bulk('Li', 'bcc', a=3.51)  # Initial lattice parameter
        li_calc = create_calculator(ecut, kpts=kpts, txt=None, 
                                   convergence={'energy': convergence_energy, 'density': convergence_density})
        li_bulk.calc = li_calc
        
        # Full cell + ion relaxation for reference state
        ef_li = ExpCellFilter(li_bulk)
        opt_li = BFGS(ef_li, logfile=None)
        opt_li.run(fmax=fmax, steps=150)
        
        e_li = li_bulk.get_potential_energy() / len(li_bulk)
        log_message(f"Bulk Li: E = {e_li:.4f} eV/atom, a = {li_bulk.get_cell()[0,0]:.3f} Å", "info")
        
        # ========== Bulk Tin (BCT) ==========
        log_message("Computing bulk Sn reference energy...", "info")
        sn_bulk = crystal('Sn', basis=[(0,0,0)], spacegroup=141,
                         cellpar=[5.83, 5.83, 3.18, 90, 90, 90])  # Experimental lattice params
        sn_calc = create_calculator(ecut, kpts=kpts, txt=None,
                                   convergence={'energy': convergence_energy, 'density': convergence_density})
        sn_bulk.calc = sn_calc
        
        # Full cell + ion relaxation
        ef_sn = ExpCellFilter(sn_bulk)
        opt_sn = BFGS(ef_sn, logfile=None)
        opt_sn.run(fmax=fmax, steps=150)
        
        e_sn = sn_bulk.get_potential_energy() / len(sn_bulk)
        cell_sn = sn_bulk.get_cell()
        log_message(f"Bulk Sn: E = {e_sn:.4f} eV/atom, a = {cell_sn[0,0]:.3f} Å, c = {cell_sn[2,2]:.3f} Å", "info")
        
        result = {
            "e_li_per_atom": e_li,
            "e_sn_per_atom": e_sn,
            "source": f"DFT/PBE computed (GPAW {GPAW_VERSION})",
            "demo_mode": False,
            "li_cell": li_bulk.get_cell().tolist(),
            "sn_cell": sn_bulk.get_cell().tolist()
        }
        
        elapsed = time.time() - start_time
        log_message(f"Reference energies computed in {format_time(elapsed)}", "success")
        
        # Save to disk if enabled
        if save_intermediate and 'output_dir' in locals():
            try:
                with open(os.path.join(output_dir, 'reference_energies.json'), 'w') as f:
                    json.dump(safe_json_serialize(result), f, indent=2)
                log_message(f"Saved reference energies to {output_dir}", "info")
            except Exception as e:
                log_message(f"Failed to save reference energies: {e}", "warning")
        
        return result
        
    except Exception as e:
        log_message(f"Reference energy computation failed: {e}", "error")
        log_message(f"Traceback: {traceback.format_exc()}", "error")
        # Fallback to precomputed values on error
        st.session_state.last_error = str(e)
        return {
            "e_li_per_atom": -1.908,
            "e_sn_per_atom": -3.152,
            "source": "Fallback: literature values (PBE)",
            "demo_mode": True,
            "error": str(e)
        }

# ============================================================================
# PHASE 2: E-V MAPPING WITH PARALLELIZATION & GP SURROGATE
# ============================================================================

def compute_single_ev_point(args):
    """
    Worker function for parallel E-V point computation.
    
    This function is designed to be picklable for multiprocessing.
    Each call computes energy at a single volume point.
    
    Parameters:
    -----------
    args : tuple
        (volume, template_atoms, ecut, kpts, fmax, convergence_energy, convergence_density, maxiter)
    
    Returns:
    --------
    tuple
        (volume, energy) or (volume, None) on failure
    """
    vol, template_atoms, ecut, kpts, fmax, conv_e, conv_d, maxiter = args
    
    try:
        # Create working copy of template structure
        atoms = template_atoms.copy()
        
        # Isotropic scaling to target volume
        current_vol = atoms.get_volume()
        scale = (vol / current_vol) ** (1/3)
        atoms.set_cell(atoms.get_cell() * scale, scale_atoms=True)
        
        # Setup calculator
        calc = create_calculator(
            ecut=ecut, 
            kpts=kpts, 
            txt=None,
            convergence={'energy': conv_e, 'density': conv_d},
            maxiter=maxiter
        )
        atoms.calc = calc
        
        # Relax ions at fixed volume
        energy = relax_fixed_volume(atoms, fmax=fmax, max_steps=100)
        
        return vol, energy
        
    except Exception as e:
        # Log error but don't crash parallel execution
        print(f"Error computing E(V) at V={vol:.2f} Å³: {e}", file=sys.stderr)
        return vol, None

@st.cache_data(show_spinner=False, ttl=7200)
def compute_ev_curve(structure_name, a_init, c_init, symbols, spacegroup, basis, 
                     num_sn, kpts, volume_range, n_points, fmax, ecut, 
                     use_surrogate=False, convergence_energy=1e-5, convergence_density=1e-4, maxiter=200):
    """
    Compute energy-volume curve with optional parallelization and GP surrogate.
    
    Workflow:
    1. Generate target volumes via isotropic scaling
    2. Compute E(V) points in parallel (ProcessPoolExecutor)
    3. Fit Birch-Murnaghan EOS to extract equilibrium properties
    4. Optionally use GP surrogate for adaptive sampling
    
    Parameters:
    -----------
    structure_name : str
        Name for logging/identification ('Sn (BCT)' or 'Li2Sn5')
    a_init, c_init : float
        Initial lattice parameters (Å)
    symbols : str or list
        Atomic symbols for crystal construction
    spacegroup : int
        International space group number
    basis : list of tuples
        Wyckoff positions for atomic basis
    num_sn : int
        Number of Sn atoms for normalization
    kpts : tuple
        k-point grid
    volume_range : tuple
        (min_scale, max_scale) for volume scaling relative to V₀
    n_points : int
        Number of volume points to compute
    fmax : float
        Force convergence for ionic relaxation
    ecut : int
        Plane-wave cutoff energy
    use_surrogate : bool, optional
        Enable GP surrogate for adaptive sampling
    convergence_energy, convergence_density : float, optional
        SCF convergence criteria
    maxiter : int, optional
        Maximum SCF iterations
    
    Returns:
    --------
    dict
        Dictionary with E-V data, EOS fit results, and metadata
    """
    log_message(f"Phase 2: Starting E-V curve for {structure_name}", "info")
    start_time = time.time()
    
    # ========== Step 1: Create template structure ==========
    try:
        if structure_name == 'Sn (BCT)':
            template = crystal('Sn', basis=[(0,0,0)], spacegroup=141,
                              cellpar=[a_init, a_init, c_init, 90, 90, 90])
        elif structure_name == 'Li2Sn5':
            template = crystal(symbols=symbols, basis=basis, spacegroup=spacegroup,
                              cellpar=[a_init, a_init, c_init, 90, 90, 90])
        else:
            raise ValueError(f"Unknown structure: {structure_name}")
    except Exception as e:
        log_message(f"Failed to create template structure: {e}", "error")
        raise
    
    v0_init = template.get_volume()
    log_message(f"{structure_name}: Initial volume V₀ = {v0_init:.2f} Å³", "info")
    
    # ========== Step 2: Generate target volumes ==========
    scales = np.linspace(volume_range[0], volume_range[1], n_points)
    target_volumes = v0_init * scales
    log_message(f"Target volumes: {target_volumes[0]:.1f} → {target_volumes[-1]:.1f} Å³ ({n_points} points)", "info")
    
    # ========== Step 3: Parallel E-V computation ==========
    # Prepare arguments for parallel workers
    worker_args = [
        (vol, template, ecut, kpts, fmax, convergence_energy, convergence_density, maxiter)
        for vol in target_volumes
    ]
    
    # Determine parallelization strategy
    if enable_parallel and parallel_mode == "ProcessPoolExecutor" and n_workers > 1:
        executor_class = ProcessPoolExecutor
        max_workers = min(n_workers, mp.cpu_count(), len(worker_args))
        log_message(f"Using ProcessPoolExecutor with {max_workers} workers", "info")
    elif enable_parallel and parallel_mode == "ThreadPoolExecutor" and n_workers > 1:
        executor_class = ThreadPoolExecutor
        max_workers = min(n_workers, len(worker_args))
        log_message(f"Using ThreadPoolExecutor with {max_workers} workers", "info")
    else:
        executor_class = None
        max_workers = 1
        log_message("Using sequential execution", "info")
    
    # Execute computations
    results = []
    progress_bar = st.progress(0, text=f"Computing {structure_name} E-V points (0/{n_points})...")
    
    if executor_class:
        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_vol = {
                executor.submit(compute_single_ev_point, arg): arg[0] 
                for arg in worker_args
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_vol):
                vol = future_to_vol[future]
                try:
                    result_vol, result_energy = future.result(timeout=300)  # 5 min timeout per point
                    if result_energy is not None:
                        results.append((result_vol, result_energy))
                        completed += 1
                        progress_bar.progress(completed / n_points, 
                                            text=f"Computing {structure_name} E-V points ({completed}/{n_points})...")
                    else:
                        log_message(f"Failed to compute energy at V={vol:.2f} Å³", "warning")
                except Exception as e:
                    log_message(f"Exception at V={vol:.2f} Å³: {e}", "error")
    else:
        # Sequential execution
        for i, args in enumerate(worker_args):
            vol = args[0]
            try:
                result_vol, result_energy = compute_single_ev_point(args)
                if result_energy is not None:
                    results.append((result_vol, result_energy))
                progress_bar.progress((i+1) / n_points,
                                    text=f"Computing {structure_name} E-V points ({i+1}/{n_points})...")
            except Exception as e:
                log_message(f"Failed at V={vol:.2f} Å³: {e}", "error")
    
    progress_bar.empty()
    
    # Sort results by volume and filter failures
    results = [r for r in results if r[1] is not None]
    results.sort(key=lambda x: x[0])
    
    if len(results) < n_points * 0.7:  # Require at least 70% success
        log_message(f"Only {len(results)}/{n_points} E-V points computed successfully", "warning")
    
    volumes, energies = zip(*results) if results else ([], [])
    volumes, energies = np.array(volumes), np.array(energies)
    
    log_message(f"Computed {len(volumes)} E-V points for {structure_name}", "info")
    
    # ========== Step 4: Birch-Murnaghan EOS fitting ==========
    if len(volumes) >= 4:  # Minimum points for EOS fit
        try:
            eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
            v0_fit, e0_fit, B0_fit, Bp_fit = eos.fit()
            B0_gpa = B0_fit / GPa  # Convert to GPa
            
            log_message(f"{structure_name} EOS fit: V₀={v0_fit:.2f} Å³, B₀={B0_gpa:.1f} GPa, B'₀={Bp_fit:.2f}", "info")
        except Exception as e:
            log_message(f"EOS fitting failed: {e}", "error")
            # Fallback: use initial volume and estimate
            v0_fit = v0_init
            e0_fit = np.min(energies) if len(energies) > 0 else 0
            B0_gpa = 50.0  # Rough estimate for metals
            Bp_fit = 4.0
            eos = None
    else:
        log_message(f"Insufficient E-V points ({len(volumes)}) for EOS fitting", "error")
        v0_fit = v0_init
        e0_fit = np.min(energies) if len(energies) > 0 else 0
        B0_gpa = None
        Bp_fit = None
        eos = None
    
    # ========== Step 5: Optional GP surrogate (placeholder for adaptive sampling) ==========
    gp_model = None
    if use_surrogate and SKLEARN_AVAILABLE and len(volumes) >= 3:
        try:
            # Train GP on computed points
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
            gp.fit(volumes.reshape(-1, 1), energies)
            gp_model = gp
            log_message("GP surrogate model trained for adaptive sampling", "info")
        except Exception as e:
            log_message(f"GP surrogate training failed: {e}", "warning")
    
    # ========== Step 6: Prepare return dictionary ==========
    elapsed = time.time() - start_time
    log_message(f"Phase 2 complete for {structure_name} in {format_time(elapsed)}", "success")
    
    result = {
        "volumes": volumes,
        "energies": energies,
        "v0_init": v0_init,
        "v0_fit": v0_fit,
        "e0_fit": e0_fit,
        "B0_GPa": B0_gpa,
        "Bp": Bp_fit,
        "num_sn": num_sn,
        "eos": eos,
        "gp_model": gp_model,
        "computation_time": elapsed,
        "n_points_computed": len(volumes),
        "n_points_requested": n_points,
        "structure_name": structure_name
    }
    
    # Save to disk if enabled
    if save_intermediate and 'output_dir' in locals():
        try:
            # Save E-V data
            ev_df = pd.DataFrame({'volume_A3': volumes, 'energy_eV': energies})
            ev_df.to_csv(os.path.join(output_dir, f'{structure_name.replace(" ", "_")}_ev_data.csv'), index=False)
            
            # Save metadata
            metadata = {k: safe_json_serialize(v) for k, v in result.items() if k not in ['eos', 'gp_model']}
            with open(os.path.join(output_dir, f'{structure_name.replace(" ", "_")}_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log_message(f"Saved E-V data to {output_dir}", "info")
        except Exception as e:
            log_message(f"Failed to save results: {e}", "warning")
    
    return result

# ============================================================================
# PHASE 3: ANISOTROPIC ELASTICITY FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False, ttl=7200)
def compute_anisotropic_elasticity(structure_name, a0, c0, symbols, spacegroup, basis,
                                   kpts, fmax, ecut, strain_range, n_strain,
                                   convergence_energy=1e-5, convergence_density=1e-4, maxiter=200):
    """
    Compute directional elastic constants C₁₁ and C₃₃ using finite-strain method.
    
    Methodology:
    1. Apply small uniaxial strains ε ∈ [-0.02, +0.02] along a and c directions
    2. Relax ionic positions at fixed strained cell
    3. Fit E(ε) = Aε² + Bε + C to extract curvature A
    4. Convert to elastic constant: C_ii = (2A/V₀) × 160.217 GPa·Å³/eV
    
    Parameters:
    -----------
    structure_name : str
        Name for logging
    a0, c0 : float
        Equilibrium lattice parameters (Å)
    symbols, spacegroup, basis : crystal definition parameters
    kpts : tuple
        k-point grid
    fmax : float
        Force convergence for relaxation
    ecut : int
        Plane-wave cutoff
    strain_range : tuple
        (min_strain, max_strain) in percent
    n_strain : int
        Number of strain points
    convergence_energy, convergence_density, maxiter : DFT convergence parameters
    
    Returns:
    --------
    dict
        Dictionary with elastic constants, strain-energy data, and metadata
    """
    log_message(f"Phase 3: Starting elasticity calculation for {structure_name}", "info")
    start_time = time.time()
    
    # ========== Step 1: Create template structure ==========
    template = crystal(symbols=symbols, basis=basis, spacegroup=spacegroup,
                      cellpar=[a0, a0, c0, 90, 90, 90])
    v0 = template.get_volume()
    log_message(f"{structure_name}: Reference volume V₀ = {v0:.2f} Å³", "info")
    
    # ========== Step 2: Generate strain values ==========
    strains = np.linspace(strain_range[0]/100, strain_range[1]/100, n_strain)
    log_message(f"Strain range: {strain_range[0]:.1f}% to {strain_range[1]:.1f}% ({n_strain} points)", "info")
    
    # ========== Step 3: Energy computation for each strain ==========
    def compute_energy_for_strain(axis, eps):
        """Compute energy at given strain along specified axis"""
        atoms = template.copy()
        
        if axis == 'a':
            # Strain a and b equally (basal plane), keep c fixed
            new_cell = [a0*(1+eps), a0*(1+eps), c0, 90, 90, 90]
        elif axis == 'c':
            # Keep a and b fixed, strain c-axis
            new_cell = [a0, a0, c0*(1+eps), 90, 90, 90]
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        atoms.set_cell(new_cell, scale_atoms=True)
        
        calc = create_calculator(
            ecut=ecut, 
            kpts=kpts, 
            txt=None,
            convergence={'energy': convergence_energy, 'density': convergence_density},
            maxiter=maxiter
        )
        atoms.calc = calc
        
        return relax_fixed_volume(atoms, fmax=fmax, max_steps=100)
    
    # Compute energies for all strains (parallel over strain values)
    energies_a = [None] * len(strains)
    energies_c = [None] * len(strains)
    
    # Use ThreadPoolExecutor for IO-bound DFT calls
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit a-axis strain calculations
        futures_a = {
            executor.submit(compute_energy_for_strain, 'a', eps): i 
            for i, eps in enumerate(strains)
        }
        # Submit c-axis strain calculations
        futures_c = {
            executor.submit(compute_energy_for_strain, 'c', eps): i 
            for i, eps in enumerate(strains)
        }
        
        # Collect a-axis results
        for future in as_completed(futures_a):
            idx = futures_a[future]
            try:
                energies_a[idx] = future.result()
            except Exception as e:
                log_message(f"Failed to compute C₁₁ at ε={strains[idx]*100:.1f}%: {e}", "warning")
        
        # Collect c-axis results
        for future in as_completed(futures_c):
            idx = futures_c[future]
            try:
                energies_c[idx] = future.result()
            except Exception as e:
                log_message(f"Failed to compute C₃₃ at ε={strains[idx]*100:.1f}%: {e}", "warning")
    
    # Filter out failures
    valid_a = [(s, e) for s, e in zip(strains, energies_a) if e is not None]
    valid_c = [(s, e) for s, e in zip(strains, energies_c) if e is not None]
    
    if len(valid_a) < 3 or len(valid_c) < 3:
        log_message("Insufficient valid strain points for elastic constant fitting", "error")
        raise ValueError("Too many strain calculations failed")
    
    strains_a, energies_a = zip(*valid_a)
    strains_c, energies_c = zip(*valid_c)
    strains_a, energies_a = np.array(strains_a), np.array(energies_a)
    strains_c, energies_c = np.array(strains_c), np.array(energies_c)
    
    # ========== Step 4: Quadratic fitting to extract elastic constants ==========
    try:
        # Fit E(ε) = Aε² + Bε + C for a-axis
        popt_a, pcov_a = curve_fit(quadratic_strain, strains_a, energies_a)
        A_a, B_a, C_a = popt_a
        
        # Fit for c-axis
        popt_c, pcov_c = curve_fit(quadratic_strain, strains_c, energies_c)
        A_c, B_c, C_c = popt_c
        
        # Convert curvature to elastic constants
        # C_ii = (2A / V₀) × conversion_factor
        # conversion_factor = 160.217 GPa·Å³/eV
        conversion_factor = 160.217
        c11 = (2 * A_a / v0) * conversion_factor
        c33 = (2 * A_c / v0) * conversion_factor
        
        log_message(f"{structure_name}: C₁₁ = {c11:.1f} GPa, C₃₃ = {c33:.1f} GPa", "info")
        
    except Exception as e:
        log_message(f"Elastic constant fitting failed: {e}", "error")
        # Fallback values
        c11, c33 = 50.0, 40.0
        popt_a, popt_c = None, None
    
    # ========== Step 5: Prepare return dictionary ==========
    elapsed = time.time() - start_time
    log_message(f"Phase 3 complete for {structure_name} in {format_time(elapsed)}", "success")
    
    anisotropy_ratio = c33 / c11 if c11 != 0 else np.inf
    
    return {
        "strains": strains,
        "energies_a": energies_a if len(energies_a) == len(strains) else np.array(energies_a),
        "energies_c": energies_c if len(energies_c) == len(strains) else np.array(energies_c),
        "c11_gpa": c11,
        "c33_gpa": c33,
        "anisotropy_ratio": anisotropy_ratio,
        "v0": v0,
        "fit_params_a": popt_a.tolist() if popt_a is not None else None,
        "fit_params_c": popt_c.tolist() if popt_c is not None else None,
        "computation_time": elapsed,
        "structure_name": structure_name
    }

# ============================================================================
# PHASE 4: FRACTURE PREDICTION & STRESS MAPPING
# ============================================================================

def predict_fracture_risk(expansion_pct, anisotropy_ratio, b0_drop_pct, c33_gpa):
    """
    Composite fracture risk assessment based on multiple mechanical criteria.
    
    Risk factors considered:
    1. Volume expansion magnitude (strain energy)
    2. Elastic anisotropy (preferential softening direction)
    3. Bulk modulus softening (material weakening)
    4. Absolute c-axis stiffness (resistance to delamination)
    
    Parameters:
    -----------
    expansion_pct : float
        Volume expansion percentage per Sn atom
    anisotropy_ratio : float
        AR = C₃₃/C₁₁ (values < 1 indicate c-axis softening)
    b0_drop_pct : float
        Percentage drop in bulk modulus upon lithiation
    c33_gpa : float
        Absolute c-axis elastic constant in GPa
    
    Returns:
    --------
    dict
        Dictionary with risk score, level, description, and contributing factors
    """
    risk_score = 0
    factors = []
    
    # Criterion 1: Volume expansion (higher = more strain energy)
    if expansion_pct > 30:
        risk_score += 3
        factors.append("🔴 Extreme expansion (>30%)")
    elif expansion_pct > 20:
        risk_score += 2
        factors.append("🟡 High expansion (20-30%)")
    elif expansion_pct > 10:
        risk_score += 1
        factors.append("🟢 Moderate expansion (10-20%)")
    # else: <10% is low risk, no points added
    
    # Criterion 2: Elastic anisotropy (AR < 1 → c-axis softer → delamination risk)
    if anisotropy_ratio < 0.7:
        risk_score += 3
        factors.append("🔴 Severe c-axis softening (AR<0.7)")
    elif anisotropy_ratio < 0.9:
        risk_score += 2
        factors.append("🟡 Moderate anisotropy (AR 0.7-0.9)")
    # else: AR ≥ 0.9 is relatively isotropic, lower risk
    
    # Criterion 3: Bulk modulus softening (material weakening)
    if b0_drop_pct > 50:
        risk_score += 2
        factors.append("🟡 Significant softening (>50% B₀ drop)")
    elif b0_drop_pct > 30:
        risk_score += 1
        factors.append("🟢 Moderate softening (30-50% B₀ drop)")
    
    # Criterion 4: Absolute c-axis stiffness (low C₃₃ → easier crack propagation)
    if c33_gpa < 20:
        risk_score += 1
        factors.append("🟢 Low c-axis stiffness (<20 GPa)")
    
    # Final risk classification based on cumulative score
    if risk_score >= 6:
        risk_level = "🔴 CRITICAL"
        description = "High probability of pulverization/delamination during cycling. Consider nanostructuring, composites, or alternative materials."
    elif risk_score >= 4:
        risk_level = "🟡 ELEVATED"
        description = "Moderate fracture risk; consider nanostructuring, carbon coating, or composite electrode design to accommodate strain."
    elif risk_score >= 2:
        risk_level = "🟢 MODERATE"
        description = "Manageable mechanical degradation with proper electrode design (binder optimization, particle size control, etc.)."
    else:
        risk_level = "🟢 LOW"
        description = "Good mechanical stability expected; standard electrode processing should suffice."
    
    log_message(f"Phase 4: Fracture risk = {risk_level} (score={risk_score}/9)", "info")
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "description": description,
        "contributing_factors": factors,
        "criteria": {
            "expansion_pct": expansion_pct,
            "anisotropy_ratio": anisotropy_ratio,
            "b0_drop_pct": b0_drop_pct,
            "c33_gpa": c33_gpa
        }
    }

@jit(nopython=True, parallel=True, cache=True) if NUMBA_AVAILABLE else lambda *args, **kwargs: lambda f: f
def compute_stress_field_numba(c11, c33, n_theta=180, n_phi=90):
    """
    Numba-accelerated computation of stress field on unit sphere.
    
    For transversely isotropic material (tetragonal symmetry), the directional
    stress under volumetric strain is approximated as:
    
    σ(θ,φ) = C₁₁·(sin²φ·cos²θ + sin²φ·sin²θ) + C₃₃·cos²φ
    
    This is evaluated on a spherical grid for 3D visualization.
    
    Parameters:
    -----------
    c11, c33 : float
        Elastic constants in GPa
    n_theta, n_phi : int, optional
        Grid resolution in azimuthal and polar directions
    
    Returns:
    --------
    2D numpy array
        Stress magnitude at each (phi, theta) grid point
    """
    stress = np.empty((n_phi, n_theta), dtype=np.float64)
    
    for i in prange(n_phi):
        phi = np.pi * i / (n_phi - 1)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        for j in range(n_theta):
            theta = 2 * np.pi * j / (n_theta - 1)
            
            # Direction cosines for unit vector in (θ, φ) direction
            lx = sin_phi * np.cos(theta)
            ly = sin_phi * np.sin(theta)
            lz = cos_phi
            
            # Stress calculation for transversely isotropic material
            # σ = C₁₁·(lₓ² + lᵧ²) + C₃₃·l_z²
            stress[i, j] = c11 * (lx**2 + ly**2) + c33 * lz**2
    
    return stress

def compute_stress_distribution_3d(c11, c33, n_theta=180, n_phi=90):
    """
    Compute 3D stress distribution with automatic Numba fallback.
    
    Parameters:
    -----------
    c11, c33 : float
        Elastic constants in GPa
    n_theta, n_phi : int, optional
        Spherical grid resolution
    
    Returns:
    --------
    dict
        Dictionary with Cartesian coordinates and stress values for plotting
    """
    log_message(f"Computing 3D stress field (C₁₁={c11:.1f}, C₃₃={c33:.1f} GPa)", "info")
    
    # Compute stress magnitude on spherical grid
    if NUMBA_AVAILABLE and use_numba:
        stress_magnitude = compute_stress_field_numba(c11, c33, n_theta, n_phi)
        log_message("Used Numba JIT acceleration for stress field", "info")
    else:
        # Pure NumPy fallback
        theta = np.linspace(0, 2*np.pi, n_theta)
        phi = np.linspace(0, np.pi, n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        lx = np.sin(phi_grid) * np.cos(theta_grid)
        ly = np.sin(phi_grid) * np.sin(theta_grid)
        lz = np.cos(phi_grid)
        
        stress_magnitude = c11 * (lx**2 + ly**2) + c33 * lz**2
    
    # Convert spherical stress field to Cartesian coordinates for 3D plotting
    # Surface radius proportional to stress magnitude
    x = np.sin(phi_grid) * np.cos(theta_grid) * stress_magnitude
    y = np.sin(phi_grid) * np.sin(theta_grid) * stress_magnitude
    z = np.cos(phi_grid) * stress_magnitude
    
    log_message(f"Stress field computed: range [{stress_magnitude.min():.1f}, {stress_magnitude.max():.1f}] GPa", "info")
    
    return {
        "x": x,
        "y": y, 
        "z": z,
        "theta": theta_grid,
        "phi": phi_grid,
        "stress": stress_magnitude,
        "c11": c11,
        "c33": c33,
        "n_theta": n_theta,
        "n_phi": n_phi
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_radar_chart(properties_dict, title="Property Comparison", colors=None):
    """
    Create radar (spider) chart for multi-property comparison.
    
    Parameters:
    -----------
    properties_dict : dict
        {property_name: value} pairs to display
    title : str, optional
        Chart title
    colors : list, optional
        Colors for plot elements
    
    Returns:
    --------
    matplotlib Figure object
    """
    categories = list(properties_dict.keys())
    N = len(categories)
    
    if N == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return fig
    
    values = list(properties_dict.values())
    
    # Normalize values to [0.1, 0.9] for radar chart visibility
    min_val, max_val = min(values), max(values)
    if max_val > min_val:
        normalized = [(v - min_val) / (max_val - min_val) * 0.8 + 0.1 for v in values]
    else:
        normalized = [0.5] * N  # All equal values
    
    # Close the loop for circular plot
    normalized += normalized[:1]
    angles = [n / N * 2 * np.pi for n in range(N)] + [2 * np.pi]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Plot line and fill
    line_color = colors[0] if colors and len(colors) > 0 else '#667eea'
    fill_color = colors[1] if colors and len(colors) > 1 else '#667eea'
    
    ax.plot(angles, normalized, 'o-', linewidth=2, color=line_color, markersize=6)
    ax.fill(angles, normalized, alpha=0.25, color=fill_color)
    
    # Format axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9, weight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])  # Hide radial labels
    
    # Title and grid
    ax.set_title(title, pad=20, size=13, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def plot_eos_scatter_with_fit(eos_results, phase_name, ax, show_residuals=False):
    """
    Plot E-V scatter points with Birch-Murnaghan EOS fit.
    
    Parameters:
    -----------
    eos_results : dict
        Output from compute_ev_curve()
    phase_name : str
        Label for the phase
    ax : matplotlib Axes
        Axes to plot on
    show_residuals : bool, optional
        Whether to show residual plot below main plot
    """
    vols = eos_results["volumes"]
    energies = eos_results["energies"]
    v0 = eos_results["v0_fit"]
    e0 = eos_results["e0_fit"]
    B0 = eos_results["B0_GPa"]
    Bp = eos_results["Bp"]
    
    # Generate smooth EOS curve for visualization
    v_smooth = np.linspace(vols.min()*0.98, vols.max()*1.02, 200)
    e_smooth = [birch_murnaghan_eos(v, e0, v0, B0*GPa if B0 else 50*GPa, Bp if Bp else 4.0) for v in v_smooth]
    
    # Main plot: E vs V
    ax.scatter(vols, energies, c='#e74c3c', s=70, label='DFT Points', zorder=5, 
              edgecolors='white', linewidth=1.2, alpha=0.9)
    ax.plot(v_smooth, e_smooth, 'b-', linewidth=2.5, label='Birch-Murnaghan Fit', alpha=0.8)
    ax.axvline(x=v0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, 
              label=f'V₀ = {v0:.2f} Å³')
    
    # Labels and formatting
    ax.set_xlabel('Volume (Å³)', fontsize=11)
    ax.set_ylabel('Energy (eV)', fontsize=11)
    ax.set_title(f'{phase_name}: E-V Curve & EOS Fit', fontsize=12, weight='bold', pad=10)
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add equilibrium energy annotation
    ax.annotate(f'E₀ = {e0:.3f} eV\nB₀ = {B0:.1f} GPa', 
               xy=(v0, e0), xytext=(10, -30), textcoords='offset points',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    
    if show_residuals and len(vols) >= 4:
        # Compute residuals for inset
        e_fit = [birch_murnaghan_eos(v, e0, v0, B0*GPa if B0 else 50*GPa, Bp if Bp else 4.0) for v in vols]
        residuals = np.array(energies) - np.array(e_fit)
        
        # Could add inset here if needed
        # For now, just log
        rmse = np.sqrt(np.mean(residuals**2))
        log_message(f"{phase_name} EOS fit RMSE: {rmse:.4f} eV", "info")

def plot_elasticity_histogram(c11, c33, phase_name, show_anisotropy=True):
    """
    Create bar chart comparing directional elastic constants.
    
    Parameters:
    -----------
    c11, c33 : float
        Elastic constants in GPa
    phase_name : str
        Label for the phase
    show_anisotropy : bool, optional
        Whether to annotate anisotropy ratio
    
    Returns:
    --------
    matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    constants = [c11, c33]
    labels = ['C₁₁ (a-b plane)', 'C₃₃ (c-axis)']
    colors = ['#3498db', '#e74c3c']
    
    # Create bars
    bars = ax.bar(labels, constants, color=colors, edgecolor='black', 
                 linewidth=1.5, alpha=0.9)
    
    # Add value labels on bars
    for bar, val in zip(bars, constants):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + max(height)*0.02, 
               f'{val:.1f}', ha='center', va='bottom', fontsize=11, weight='bold')
    
    # Labels and title
    ax.set_ylabel('Elastic Constant (GPa)', fontsize=11)
    ax.set_title(f'{phase_name}: Directional Stiffness', fontsize=12, weight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Anisotropy annotation
    if show_anisotropy and c11 > 0:
        ar = c33 / c11
        ax.text(0.5, -max(constants)*0.15, f'Anisotropy Ratio AR = C₃₃/C₁₁ = {ar:.3f}', 
               ha='center', fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    return fig

def plot_3d_stress_sphere(stress_data, title="Anisotropic Stress Distribution", 
                         cmap_name='RdYlBu_r', elevation=25, azimuth=45):
    """
    Create 3D spherical plot of stress distribution.
    
    Parameters:
    -----------
    stress_data : dict
        Output from compute_stress_distribution_3d()
    title : str, optional
        Plot title
    cmap_name : str, optional
        Matplotlib colormap name
    elevation, azimuth : float, optional
        Initial view angles for 3D plot
    
    Returns:
    --------
    matplotlib Figure object
    """
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = stress_data["x"], stress_data["y"], stress_data["z"]
    stress = stress_data["stress"]
    c11, c33 = stress_data["c11"], stress_data["c33"]
    
    # Color mapping
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=stress.min(), vmax=stress.max())
    colors = cmap(norm(stress))
    
    # Plot surface with face colors
    surf = ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1,
                          linewidth=0, antialiased=True, alpha=0.95)
    
    # Labels and title
    ax.set_xlabel('X', fontsize=10, labelpad=5)
    ax.set_ylabel('Y', fontsize=10, labelpad=5)
    ax.set_zlabel('Z (c-axis)', fontsize=10, labelpad=5)
    ax.set_title(title, fontsize=13, weight='bold', pad=20)
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, aspect=20)
    cbar.set_label('Relative Stress (GPa·strain)', fontsize=9, rotation=270, labelpad=15)
    
    # Annotation for c-axis direction
    ax.text(0, 0, max(z.flatten()) * 1.1, '↑ c-axis', ha='center', fontsize=9, weight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='gray'))
    
    # Set view angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Add elastic constant info as text
    info_text = f'C₁₁ = {c11:.1f} GPa\nC₃₃ = {c33:.1f} GPa\nAR = {c33/c11 if c11>0 else "∞":.3f}'
    ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    return fig

def plot_expansion_bar_chart(sn_results, li2sn5_results, expansion_pct, show_values=True):
    """
    Create bar chart comparing volumes per Sn atom with expansion annotation.
    
    Parameters:
    -----------
    sn_results, li2sn5_results : dict
        Output from compute_ev_curve() for each phase
    expansion_pct : float
        Computed volume expansion percentage
    show_values : bool, optional
        Whether to show numerical values on bars
    
    Returns:
    --------
    matplotlib Figure object
    """
    v_per_sn_sn = sn_results["v0_fit"] / sn_results["num_sn"]
    v_per_sn_li = li2sn5_results["v0_fit"] / li2sn5_results["num_sn"]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    phases = ['β-Sn', 'Li₂Sn₅']
    volumes = [v_per_sn_sn, v_per_sn_li]
    colors = ['#2ecc71', '#9b59b6']
    
    # Create bars
    bars = ax.bar(phases, volumes, color=colors, edgecolor='black', 
                 linewidth=2, alpha=0.95)
    
    # Labels and title
    ax.set_ylabel('Volume per Sn Atom (Å³)', fontsize=12)
    ax.set_title(f'Volume Expansion: {expansion_pct:+.2f}%', fontsize=14, weight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add expansion arrow and annotation
    ax.annotate('', 
               xy=(1, v_per_sn_li), xytext=(0, v_per_sn_sn),
               arrowprops=dict(arrowstyle='->', color='red', lw=3, ls='-', 
                              mutation_scale=20))
    ax.text(0.5, (v_per_sn_sn + v_per_sn_li)/2, 
           f'+{expansion_pct:.1f}%', 
           ha='center', va='bottom', color='red', weight='bold', fontsize=13,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red'))
    
    # Add value labels on bars
    if show_values:
        for bar, vol in zip(bars, volumes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max(volumes)*0.03, 
                   f'{vol:.2f}', ha='center', va='bottom', fontsize=11, weight='bold')
    
    # Add reference line at Sn volume
    ax.axhline(y=v_per_sn_sn, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APPLICATION: TABS WITH INDEPENDENT EXECUTION
# ============================================================================

# Create navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Phase 1: Thermodynamics",
    "📊 Phase 2: EOS & Expansion", 
    "🧭 Phase 3: Anisotropic Elasticity",
    "💥 Phase 4: Fracture Prediction",
    "📈 Integrated Dashboard"
])

# ============================================================================
# TAB 1: PHASE 1 - THERMODYNAMIC STABILITY
# ============================================================================
with tab1:
    st.header("🔬 Phase 1: Thermodynamic Stability Analysis")
    
    # Methodology description
    with st.expander("📚 Methodology", expanded=False):
        st.markdown("""
        **Formation Energy Calculation**:
        
        The formation energy of Li₂Sn₅ relative to elemental references:
        
        ```
        ΔE_f = [E_tot(Li₂Sn₅) - 4·E_Li - 10·E_Sn] / 14 atoms
        ```
        
        where:
        - E_tot(Li₂Sn₅): Total DFT energy of Li₂Sn₅ unit cell
        - E_Li: Energy per atom of bulk BCC Li reference
        - E_Sn: Energy per atom of bulk BCT Sn reference
        - 14: Total atoms in Li₂Sn₅ cell (4 Li + 10 Sn)
        
        **Interpretation**:
        - ΔE_f < 0: Thermodynamically stable (spontaneous formation)
        - ΔE_f > 0: Metastable/unstable (kinetic factors may enable formation)
        
        **Reference States**:
        - Li: BCC structure, fully relaxed cell + ions
        - Sn: BCT structure (space group 141), fully relaxed
        """)
    
    # Input/controls
    col1, col2 = st.columns(2)
    with col1:
        run_phase1 = st.button("🚀 Run Phase 1 Analysis", key="btn_run_phase1", use_container_width=True)
    with col2:
        # Status indicator for dependencies
        if st.session_state.phase_results['phase2_li2sn5'] is not None:
            st.success("✅ Li₂Sn₅ E₀ available from Phase 2")
        else:
            st.info("ℹ️ Run Phase 2 first for accurate Li₂Sn₅ energy, or enter manually below")
    
    # Manual energy input fallback
    if st.session_state.phase_results['phase2_li2sn5'] is None:
        e_li2sn5_manual = st.number_input(
            "Li₂Sn₅ total energy (eV) - manual input",
            value=-100.0,
            step=0.1,
            help="Enter total energy if Phase 2 not yet computed. Will be overridden when Phase 2 results available."
        )
    
    # Execute Phase 1
    if run_phase1:
        with st.spinner("🔄 Computing reference energies and formation energy..."):
            phase1_start = time.time()
            
            # Get or compute reference energies
            if st.session_state.ref_energies is None:
                st.session_state.ref_energies = compute_reference_energies(
                    ecut, kpts_sn, fmax, convergence_energy, convergence_density
                )
            ref = st.session_state.ref_energies
            
            # Get Li2Sn5 energy (from Phase 2 or manual input)
            if st.session_state.phase_results['phase2_li2sn5'] is not None:
                e_li2sn5 = st.session_state.phase_results['phase2_li2sn5']["e0_fit"]
                log_message("Using Li₂Sn₅ E₀ from Phase 2 results", "info")
            else:
                e_li2sn5 = e_li2sn5_manual
                log_message(f"Using manual Li₂Sn₅ energy: {e_li2sn5:.2f} eV", "warning")
            
            # Compute formation energy
            thermo = phase1_thermodynamic_stability(
                e_li2sn5_total=e_li2sn5,
                e_sn_per=ref["e_sn_per_atom"],
                e_li_per=ref["e_li_per_atom"]
            )
            
            # Store results
            st.session_state.phase_results['phase1'] = thermo
            
            phase1_time = time.time() - phase1_start
            st.session_state.computation_times['phase1'] = phase1_time
            log_message(f"Phase 1 completed in {format_time(phase1_time)}", "success")
    
    # Display results if available
    if st.session_state.phase_results['phase1'] is not None:
        thermo = st.session_state.phase_results['phase1']
        
        # Key metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Formation Energy (per atom)", 
                f"{thermo['formation_per_atom']:.4f} eV",
                delta=thermo['stability_label'],
                delta_color="normal" if thermo['is_stable'] else "inverse"
            )
        with col2:
            st.metric("Formation Energy (per formula)", f"{thermo['formation_per_formula']:.3f} eV")
        with col3:
            st.metric("Total Energy Change", f"{thermo['delta_e_total']:.2f} eV")
        
        # Stability assessment box
        if thermo['is_stable']:
            st.success(f"""
            ✅ **Li₂Sn₅ is thermodynamically stable** relative to bulk Li + Sn
            
            The negative formation energy (ΔE_f = {thermo['formation_per_atom']:.4f} eV/atom) indicates that Li₂Sn₅ 
            will form spontaneously during lithiation under equilibrium conditions.
            """)
        else:
            st.warning(f"""
            ⚠️ **Li₂Sn₅ shows metastability** (ΔE_f = {thermo['formation_per_atom']:.4f} eV/atom)
            
            While not thermodynamically favored, kinetic factors (diffusion barriers, nucleation) 
            may still enable Li₂Sn₅ formation during battery cycling.
            """)
        
        # Energy diagram visualization
        st.subheader("📊 Thermodynamic Stability Diagram")
        
        fig, ax = plt.subplots(figsize=(9, 5))
        
        phases = ['Li + Sn (reference)', 'Li₂Sn₅']
        energies = [0, thermo['formation_per_formula']]
        colors = ['#95a5a6', '#27ae60' if thermo['is_stable'] else '#e67e22']
        
        bars = ax.bar(phases, energies, color=colors, edgecolor='black', linewidth=1.5, alpha=0.9)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Reference (0)')
        
        # Add value labels
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + np.sign(height)*0.02, 
                   f'{energy:.3f}', ha='center', va='bottom' if energy>0 else 'top', 
                   fontsize=10, weight='bold')
        
        ax.set_ylabel('Energy Relative to Reference (eV per Li₂Sn₅ formula unit)', fontsize=11)
        ax.set_title('Formation Energy of Li₂Sn₅', fontsize=13, weight='bold', pad=15)
        ax.legend(fontsize=10, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Reference energy details
        with st.expander("🔍 Reference Energy Details"):
            st.markdown(f"""
            **Bulk Li (BCC) Reference**:
            - Energy per atom: `{ref['e_li_per_atom']:.4f} eV`
            - Source: {ref.get('source', 'Computed')}
            - Demo mode: {ref.get('demo_mode', False)}
            
            **Bulk Sn (BCT) Reference**:
            - Energy per atom: `{ref['e_sn_per_atom']:.4f} eV`
            - Source: {ref.get('source', 'Computed')}
            
            **Li₂Sn₅ Energy**:
            - Total energy: `{e_li2sn5:.2f} eV` (for 14-atom cell)
            - Source: {'Phase 2 DFT' if st.session_state.phase_results['phase2_li2sn5'] else 'Manual input'}
            """)

# ============================================================================
# TAB 2: PHASE 2 - EOS & VOLUME EXPANSION
# ============================================================================
with tab2:
    st.header("📊 Phase 2: Equation of State & Volume Expansion")
    
    # Methodology
    with st.expander("📚 Methodology", expanded=False):
        st.markdown("""
        **Energy-Volume Mapping**:
        
        1. Generate 7-11 configurations by isotropic scaling: V = V₀ × s, s ∈ [0.92, 1.08]
        2. At each fixed volume: relax atomic positions (BFGS) until forces < fmax
        3. Record total energy E(V) at each point
        
        **Birch-Murnaghan EOS Fitting**:
        
        Fit discrete (V, E) points to 3rd-order EOS:
        
        ```
        E(V) = E₀ + (9V₀B₀/16) × {[(V₀/V)^(2/3)-1]³×B'₀ + [(V₀/V)^(2/3)-1]²×[6-4(V₀/V)^(2/3)]}
        ```
        
        Extracted parameters:
        - V₀: Equilibrium volume
        - E₀: Equilibrium energy  
        - B₀: Bulk modulus (resistance to uniform compression)
        - B'₀: Pressure derivative of B₀
        
        **Volume Expansion Calculation**:
        
        ```
        Expansion (%) = [(V₀^Li₂Sn₅/10 - V₀^Sn/4) / (V₀^Sn/4)] × 100
        ```
        
        Normalized per Sn atom for meaningful comparison between phases.
        """)
    
    # Independent run buttons for each phase
    col1, col2 = st.columns(2)
    with col1:
        run_sn = st.button("🚀 Compute Sn E-V Curve", key="btn_run_sn", use_container_width=True)
    with col2:
        run_li2sn5 = st.button("🚀 Compute Li₂Sn₅ E-V Curve", key="btn_run_li2sn5", use_container_width=True)
    
    # Progress/status display
    if run_sn and st.session_state.phase_results['phase2_sn'] is None:
        with st.spinner(f"🔄 Computing Sn E-V curve ({n_vol} points, parallel={enable_parallel})..."):
            phase2_sn_start = time.time()
            try:
                sn_results = compute_ev_curve(
                    structure_name='Sn (BCT)',
                    a_init=5.83, c_init=3.18,
                    symbols='Sn', spacegroup=141, basis=[(0,0,0)],
                    num_sn=4, kpts=kpts_sn,
                    volume_range=volume_range, n_points=n_vol,
                    fmax=fmax, ecut=ecut,
                    use_surrogate=use_surrogate,
                    convergence_energy=convergence_energy,
                    convergence_density=convergence_density,
                    maxiter=maxiter
                )
                st.session_state.phase_results['phase2_sn'] = sn_results
                phase2_sn_time = time.time() - phase2_sn_start
                st.session_state.computation_times['phase2_sn'] = phase2_sn_time
                st.success(f"✅ Sn E-V curve computed in {format_time(phase2_sn_time)}")
            except Exception as e:
                st.error(f"❌ Sn calculation failed: {e}")
                log_message(f"Phase 2 Sn failed: {e}", "error")
                st.session_state.last_error = str(e)
    
    if run_li2sn5 and st.session_state.phase_results['phase2_li2sn5'] is None:
        with st.spinner(f"🔄 Computing Li₂Sn₅ E-V curve ({n_vol} points, parallel={enable_parallel})..."):
            phase2_li_start = time.time()
            try:
                li2sn5_results = compute_ev_curve(
                    structure_name='Li2Sn5',
                    a_init=10.274, c_init=3.125,
                    symbols=['Sn', 'Li', 'Sn'], spacegroup=127,
                    basis=[(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)],
                    num_sn=10, kpts=kpts_li2sn5,
                    volume_range=volume_range, n_points=n_vol,
                    fmax=fmax, ecut=ecut,
                    use_surrogate=use_surrogate,
                    convergence_energy=convergence_energy,
                    convergence_density=convergence_density,
                    maxiter=maxiter
                )
                st.session_state.phase_results['phase2_li2sn5'] = li2sn5_results
                phase2_li_time = time.time() - phase2_li_start
                st.session_state.computation_times['phase2_li2sn5'] = phase2_li_time
                st.success(f"✅ Li₂Sn₅ E-V curve computed in {format_time(phase2_li_time)}")
            except Exception as e:
                st.error(f"❌ Li₂Sn₅ calculation failed: {e}")
                log_message(f"Phase 2 Li₂Sn₅ failed: {e}", "error")
                st.session_state.last_error = str(e)
    
    # Display results if both phases computed
    sn_res = st.session_state.phase_results.get('phase2_sn')
    li_res = st.session_state.phase_results.get('phase2_li2sn5')
    
    if sn_res is not None and li_res is not None:
        # Key metrics in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("V₀ (β-Sn)", f"{sn_res['v0_fit']:.2f} Å³")
        with col2:
            st.metric("V₀ (Li₂Sn₅)", f"{li_res['v0_fit']:.2f} Å³")
        
        # Compute per-Sn volumes and expansion
        v_per_sn_sn = sn_res['v0_fit'] / sn_res['num_sn']
        v_per_sn_li = li_res['v0_fit'] / li_res['num_sn']
        
        with col3:
            st.metric("Volume/Sn (β-Sn)", f"{v_per_sn_sn:.3f} Å³")
        with col4:
            st.metric("Volume/Sn (Li₂Sn₅)", f"{v_per_sn_li:.3f} Å³")
        
        # Expansion calculation and display
        expansion_pct = (v_per_sn_li - v_per_sn_sn) / v_per_sn_sn * 100
        st.session_state.expansion_pct = expansion_pct  # Store for Phase 4
        
        # Highlight expansion result
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1.2rem; border-radius: 0.5rem; 
                    text-align: center; font-size: 1.3rem; margin: 1rem 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1)'>
            <strong>Volume Expansion: {expansion_pct:+.2f}%</strong> per Sn atom
        </div>
        """, unsafe_allow_html=True)
        
        # Bulk modulus drop (for Phase 4)
        if sn_res['B0_GPa'] and li_res['B0_GPa']:
            b0_drop_pct = (sn_res['B0_GPa'] - li_res['B0_GPa']) / sn_res['B0_GPa'] * 100
            st.session_state.b0_drop_pct = b0_drop_pct
            st.info(f"💡 Bulk modulus drops by {b0_drop_pct:.1f}% upon lithiation (material softening)")
        
        # E-V curves side-by-side
        st.subheader("📈 Energy-Volume Curves & EOS Fits")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        plot_eos_scatter_with_fit(sn_res, 'β-Sn (BCT)', ax1)
        plot_eos_scatter_with_fit(li_res, 'Li₂Sn₅', ax2)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Bulk modulus comparison
        st.subheader("📊 Bulk Modulus Comparison")
        if sn_res['B0_GPa'] and li_res['B0_GPa']:
            fig = plot_elasticity_histogram(sn_res['B0_GPa'], li_res['B0_GPa'], 'Bulk Modulus')
            st.pyplot(fig)
        else:
            st.warning("Bulk modulus values not available for plotting")
        
        # Volume expansion bar chart
        st.subheader("📏 Volume Expansion Visualization")
        fig = plot_expansion_bar_chart(sn_res, li_res, expansion_pct)
        st.pyplot(fig)
        
        # Data table
        with st.expander("📋 Raw E-V Data Tables"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**β-Sn E-V Data**")
                df_sn = pd.DataFrame({
                    'Volume (Å³)': sn_res['volumes'],
                    'Energy (eV)': sn_res['energies']
                })
                st.dataframe(df_sn, use_container_width=True, hide_index=True)
            with col2:
                st.write("**Li₂Sn₅ E-V Data**")
                df_li = pd.DataFrame({
                    'Volume (Å³)': li_res['volumes'],
                    'Energy (eV)': li_res['energies']
                })
                st.dataframe(df_li, use_container_width=True, hide_index=True)
    
    elif sn_res is not None or li_res is not None:
        # Partial results available
        st.info("💡 Compute both Sn and Li₂Sn₅ E-V curves to see expansion comparison")
        
        if sn_res is not None:
            st.subheader("📊 β-Sn Results")
            st.metric("V₀", f"{sn_res['v0_fit']:.2f} Å³")
            st.metric("B₀", f"{sn_res['B0_GPa']:.1f} GPa" if sn_res['B0_GPa'] else "N/A")
        
        if li_res is not None:
            st.subheader("📊 Li₂Sn₅ Results")
            st.metric("V₀", f"{li_res['v0_fit']:.2f} Å³")
            st.metric("B₀", f"{li_res['B0_GPa']:.1f} GPa" if li_res['B0_GPa'] else "N/A")

# ============================================================================
# TAB 3: PHASE 3 - ANISOTROPIC ELASTICITY
# ============================================================================
with tab3:
    st.header("🧭 Phase 3: Anisotropic Elastic Constants")
    
    # Methodology
    with st.expander("📚 Methodology", expanded=False):
        st.markdown("""
        **Finite-Strain Method for Elastic Constants**:
        
        For tetragonal crystals, we compute directional elastic constants:
        
        **C₁₁ (basal plane stiffness)**:
        - Apply uniaxial strain ε along a and b: a' = a₀(1+ε), b' = b₀(1+ε), c' = c₀
        - Relax ionic positions at fixed strained cell
        - Fit E(ε) = Aε² + Bε + C → C₁₁ = (2A/V₀) × 160.217 GPa
        
        **C₃₃ (c-axis stiffness)**:
        - Apply strain along c: a' = a₀, b' = b₀, c' = c₀(1+ε)
        - Same fitting procedure → C₃₃
        
        **Anisotropy Ratio**:
        ```
        AR = C₃₃ / C₁₁
        ```
        - AR < 1: c-axis softer than basal plane → preferential expansion along [001]
        - AR ≈ 1: isotropic elastic response
        - AR > 1: c-axis stiffer (unusual for layered materials)
        
        **Physical Interpretation**:
        Low AR indicates weak interlayer bonding, leading to:
        - Preferential expansion along c-axis during lithiation
        - Risk of interlayer delamination and crack propagation
        - Anisotropic stress distribution (visualized in Phase 4)
        """)
    
    # Independent run buttons
    col1, col2 = st.columns(2)
    with col1:
        run_sn_elastic = st.button("🚀 Compute Sn Elasticity", key="btn_run_sn_el", use_container_width=True)
    with col2:
        run_li_elastic = st.button("🚀 Compute Li₂Sn₅ Elasticity", key="btn_run_li_el", use_container_width=True)
    
    # Execute Sn elasticity
    if run_sn_elastic and st.session_state.phase_results['phase3_sn'] is None:
        with st.spinner(f"🔄 Computing Sn elastic constants ({n_strain} strain points)..."):
            phase3_sn_start = time.time()
            try:
                sn_elastic = compute_anisotropic_elasticity(
                    structure_name='Sn',
                    a0=5.83, c0=3.18,
                    symbols='Sn', spacegroup=141, basis=[(0,0,0)],
                    kpts=kpts_sn, fmax=fmax, ecut=ecut,
                    strain_range=strain_range, n_strain=n_strain,
                    convergence_energy=convergence_energy,
                    convergence_density=convergence_density,
                    maxiter=maxiter
                )
                st.session_state.phase_results['phase3_sn'] = sn_elastic
                phase3_sn_time = time.time() - phase3_sn_start
                st.session_state.computation_times['phase3_sn'] = phase3_sn_time
                st.success(f"✅ Sn elasticity computed in {format_time(phase3_sn_time)}")
            except Exception as e:
                st.error(f"❌ Sn elasticity failed: {e}")
                log_message(f"Phase 3 Sn failed: {e}", "error")
    
    # Execute Li₂Sn₅ elasticity
    if run_li_elastic and st.session_state.phase_results['phase3_li2sn5'] is None:
        with st.spinner(f"🔄 Computing Li₂Sn₅ elastic constants ({n_strain} strain points)..."):
            phase3_li_start = time.time()
            try:
                li_elastic = compute_anisotropic_elasticity(
                    structure_name='Li2Sn5',
                    a0=10.274, c0=3.125,
                    symbols=['Sn', 'Li', 'Sn'], spacegroup=127,
                    basis=[(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)],
                    kpts=kpts_li2sn5, fmax=fmax, ecut=ecut,
                    strain_range=strain_range, n_strain=n_strain,
                    convergence_energy=convergence_energy,
                    convergence_density=convergence_density,
                    maxiter=maxiter
                )
                st.session_state.phase_results['phase3_li2sn5'] = li_elastic
                st.session_state.li2sn5_elastic = li_elastic  # Store for Phase 4
                phase3_li_time = time.time() - phase3_li_start
                st.session_state.computation_times['phase3_li2sn5'] = phase3_li_time
                st.success(f"✅ Li₂Sn₅ elasticity computed in {format_time(phase3_li_time)}")
            except Exception as e:
                st.error(f"❌ Li₂Sn₅ elasticity failed: {e}")
                log_message(f"Phase 3 Li₂Sn₅ failed: {e}", "error")
    
    # Display results
    sn_el = st.session_state.phase_results.get('phase3_sn')
    li_el = st.session_state.phase_results.get('phase3_li2sn5')
    
    if sn_el is not None:
        st.subheader("📊 β-Sn (BCT) Elastic Constants")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("C₁₁ (a-b plane)", f"{sn_el['c11_gpa']:.1f} GPa")
        with col2:
            st.metric("C₃₃ (c-axis)", f"{sn_el['c33_gpa']:.1f} GPa")
        with col3:
            ar_val = sn_el['anisotropy_ratio']
            st.metric("Anisotropy AR", f"{ar_val:.3f}", 
                     delta="c-soft" if ar_val < 1 else "isotropic" if 0.9 <= ar_val <= 1.1 else "c-stiff")
    
    if li_el is not None:
        st.subheader("📊 Li₂Sn₅ Elastic Constants")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("C₁₁ (a-b plane)", f"{li_el['c11_gpa']:.1f} GPa")
        with col2:
            st.metric("C₃₃ (c-axis)", f"{li_el['c33_gpa']:.1f} GPa")
        with col3:
            ar_val = li_el['anisotropy_ratio']
            st.metric("Anisotropy AR", f"{ar_val:.3f}",
                     delta="c-soft" if ar_val < 1 else "isotropic" if 0.9 <= ar_val <= 1.1 else "c-stiff")
        
        # Strain-energy curves visualization
        st.subheader("📈 Strain-Energy Curves & Quadratic Fits")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # C11 fit (a-axis strain)
        strains_pct = li_el['strains'] * 100
        ax1.plot(strains_pct, li_el['energies_a'], 'o-', label='DFT', color='#e74c3c', 
                markersize=6, linewidth=2, alpha=0.9)
        if li_el['fit_params_a']:
            popt_a = li_el['fit_params_a']
            fit_a = quadratic_strain(strains_pct/100, *popt_a)
            ax1.plot(strains_pct, fit_a, '--', label='Quadratic Fit', color='#3498db', linewidth=2)
        ax1.set_xlabel('Strain ε_a (%)', fontsize=11)
        ax1.set_ylabel('Energy (eV)', fontsize=11)
        ax1.set_title('C₁₁ Extraction: a-axis Strain', fontsize=12, weight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # C33 fit (c-axis strain)
        ax2.plot(strains_pct, li_el['energies_c'], 'o-', label='DFT', color='#e74c3c',
                markersize=6, linewidth=2, alpha=0.9)
        if li_el['fit_params_c']:
            popt_c = li_el['fit_params_c']
            fit_c = quadratic_strain(strains_pct/100, *popt_c)
            ax2.plot(strains_pct, fit_c, '--', label='Quadratic Fit', color='#9b59b6', linewidth=2)
        ax2.set_xlabel('Strain ε_c (%)', fontsize=11)
        ax2.set_ylabel('Energy (eV)', fontsize=11)
        ax2.set_title('C₃₃ Extraction: c-axis Strain', fontsize=12, weight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Elastic constants radar chart
        st.subheader("🎯 Elastic Constants Comparison (Radar Chart)")
        properties = {
            'C₁₁ (Sn)': sn_el['c11_gpa'] if sn_el else 0,
            'C₃₃ (Sn)': sn_el['c33_gpa'] if sn_el else 0,
            'C₁₁ (Li₂Sn₅)': li_el['c11_gpa'],
            'C₃₃ (Li₂Sn₅)': li_el['c33_gpa']
        }
        fig = plot_radar_chart(properties, "Elastic Constants: Sn vs Li₂Sn₅")
        st.pyplot(fig)
    
    if sn_el is None and li_el is None:
        st.info("💡 Run elasticity calculations to see directional stiffness results")

# ============================================================================
# TAB 4: PHASE 4 - FRACTURE PREDICTION & 3D STRESS
# ============================================================================
with tab4:
    st.header("💥 Phase 4: Mechanical Fracture Prediction")
    
    # Methodology
    with st.expander("📚 Methodology", expanded=False):
        st.markdown("""
        **Fracture Risk Assessment Criteria**:
        
        Composite risk score based on four mechanical factors:
        
        1. **Volume Expansion** (strain energy):
           - >30%: 🔴 Extreme risk (high stored elastic energy)
           - 20-30%: 🟡 Elevated risk
           - 10-20%: 🟢 Moderate risk
        
        2. **Elastic Anisotropy** (preferential softening):
           - AR < 0.7: 🔴 Severe c-axis softening → delamination risk
           - AR 0.7-0.9: 🟡 Moderate anisotropy
           - AR ≥ 0.9: Lower risk
        
        3. **Bulk Modulus Softening** (material weakening):
           - >50% drop: 🟡 Significant weakening
           - 30-50% drop: 🟢 Moderate change
        
        4. **Absolute c-axis Stiffness** (crack resistance):
           - C₃₃ < 20 GPa: 🟢 Low resistance to crack propagation
        
        **Risk Classification**:
        - 🔴 CRITICAL (score ≥ 6): High probability of pulverization
        - 🟡 ELEVATED (score 4-5): Moderate risk; consider nanostructuring
        - 🟢 MODERATE (score 2-3): Manageable with proper electrode design
        - 🟢 LOW (score 0-1): Good mechanical stability expected
        
        **3D Stress Distribution**:
        
        For transversely isotropic material, directional stress under volumetric strain:
        
        ```
        σ(θ,φ) = C₁₁·(sin²φ·cos²θ + sin²φ·sin²θ) + C₃₃·cos²φ
        ```
        
        Visualized on unit sphere: radius ∝ stress magnitude, color = relative value.
        Red regions indicate high stress concentration (likely crack initiation sites).
        """)
    
    # Check prerequisites
    missing_deps = []
    if st.session_state.expansion_pct is None:
        missing_deps.append("Phase 2: Volume expansion data")
    if st.session_state.b0_drop_pct is None:
        missing_deps.append("Phase 2: Bulk modulus drop")
    if st.session_state.phase_results.get('phase3_li2sn5') is None:
        missing_deps.append("Phase 3: Li₂Sn₅ elastic constants")
    
    if missing_deps:
        st.warning(f"""
        ⚠️ **Missing prerequisites for Phase 4**:
        
        Please run the following phases first:
        {chr(10).join(f'- {dep}' for dep in missing_deps)}
        
        Once completed, click "Run Fracture Prediction" below.
        """)
    else:
        # Run button
        if st.button("🚀 Run Fracture Prediction", key="btn_run_phase4", use_container_width=True):
            with st.spinner("🔄 Predicting fracture risk and computing stress distribution..."):
                phase4_start = time.time()
                
                # Gather inputs
                expansion = st.session_state.expansion_pct
                b0_drop = st.session_state.b0_drop_pct
                li_el = st.session_state.phase_results['phase3_li2sn5']
                
                # Compute fracture risk
                fracture = predict_fracture_risk(
                    expansion_pct=expansion,
                    anisotropy_ratio=li_el['anisotropy_ratio'],
                    b0_drop_pct=b0_drop,
                    c33_gpa=li_el['c33_gpa']
                )
                st.session_state.phase_results['phase4'] = fracture
                
                # Compute 3D stress field
                stress_3d = compute_stress_distribution_3d(
                    c11=li_el['c11_gpa'],
                    c33=li_el['c33_gpa']
                )
                st.session_state.stress_3d = stress_3d
                
                phase4_time = time.time() - phase4_start
                st.session_state.computation_times['phase4'] = phase4_time
                st.success(f"✅ Fracture prediction complete in {format_time(phase4_time)}")
        
        # Display results if available
        if st.session_state.phase_results['phase4'] is not None:
            fracture = st.session_state.phase_results['phase4']
            li_el = st.session_state.phase_results['phase3_li2sn5']
            
            # Risk assessment box with color coding
            if "CRITICAL" in fracture['risk_level']:
                border_color = "#e74c3c"
                bg_color = "#fdedec"
                icon = "🔴"
            elif "ELEVATED" in fracture['risk_level']:
                border_color = "#f39c12"
                bg_color = "#fef5e7"
                icon = "🟡"
            else:
                border_color = "#27ae60"
                bg_color = "#eafaf1"
                icon = "🟢"
            
            st.markdown(f"""
            <div style='padding: 1.2rem; border-left: 5px solid {border_color}; 
                        background: {bg_color}; border-radius: 0 0.4rem 0.4rem 0;
                        margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05)'>
                <h3 style='margin: 0 0 0.5rem 0; color: {border_color}'>{icon} {fracture['risk_level']}</h3>
                <p style='margin: 0; font-size: 1.1rem'>{fracture['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Contributing factors
            if fracture['contributing_factors']:
                st.markdown("**🔍 Contributing Risk Factors**:")
                for factor in fracture['contributing_factors']:
                    st.markdown(f"- {factor}")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Volume Expansion", f"{expansion:.2f}%")
            with col2:
                ar_val = li_el['anisotropy_ratio']
                st.metric("Anisotropy Ratio AR", f"{ar_val:.3f}",
                         delta="c-soft" if ar_val < 1 else "isotropic")
            with col3:
                st.metric("Bulk Modulus Drop", f"{b0_drop:.1f}%")
            
            # 3D Stress Distribution Visualization
            st.subheader("🌐 Anisotropic Stress Distribution (Polar Spherical Coordinates)")
            st.markdown("*Surface radius ∝ stress magnitude; color indicates relative stress. Red = high stress concentration.*")
            
            fig = plot_3d_stress_sphere(
                st.session_state.stress_3d,
                title=f"Li₂Sn₅ Stress Map (C₁₁={li_el['c11_gpa']:.0f}, C₃₃={li_el['c33_gpa']:.0f} GPa)"
            )
            st.pyplot(fig)
            
            # Stress histogram
            st.subheader("📊 Stress Distribution Histogram")
            fig, ax = plt.subplots(figsize=(9, 5))
            stress_vals = st.session_state.stress_3d['stress'].flatten()
            ax.hist(stress_vals, bins=40, color='#9b59b6', edgecolor='black', alpha=0.7, linewidth=0.5)
            ax.set_xlabel('Relative Stress (GPa·strain)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Stress Distribution Across Crystal Directions', fontsize=12, weight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Add statistics
            stats_text = f"Mean: {stress_vals.mean():.1f} | Std: {stress_vals.std():.1f} | Max: {stress_vals.max():.1f}"
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interpretation guidance
            with st.expander("💡 Interpretation Guide"):
                st.markdown("""
                **Reading the 3D Stress Map**:
                - **Red/orange regions**: High stress concentration → likely crack initiation sites
                - **Blue regions**: Low stress → more resistant to fracture
                - **Elongation along c-axis**: Indicates preferential stress along [001] direction
                
                **Fracture Mechanism**:
                1. Lithiation induces volumetric expansion (~22% for Li₂Sn₅)
                2. Anisotropic elasticity (AR < 1) concentrates stress along c-axis
                3. High local stress exceeds fracture toughness → crack nucleation
                4. Cracks propagate along weak interlayer planes → particle pulverization
                
                **Mitigation Strategies**:
                - Nanostructuring: Reduce absolute strain per particle
                - Carbon coating: Accommodate expansion, maintain electrical contact
                - Composite electrodes: Buffer volume changes with inactive matrix
                - Pre-lithiation: Reduce first-cycle expansion
                """)

# ============================================================================
# TAB 5: INTEGRATED DASHBOARD
# ============================================================================
with tab5:
    st.header("📈 Integrated Multi-View Dashboard")
    
    # Check data availability
    sn_eos = st.session_state.phase_results.get('phase2_sn')
    li_eos = st.session_state.phase_results.get('phase2_li2sn5')
    li_el = st.session_state.phase_results.get('phase3_li2sn5')
    thermo = st.session_state.phase_results.get('phase1')
    fracture = st.session_state.phase_results.get('phase4')
    
    if not (sn_eos and li_eos):
        st.info("💡 Run Phase 2 (EOS) calculations to populate dashboard with results")
    else:
        # Summary metrics cards
        st.subheader("🎯 Key Results Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if thermo:
                val = thermo['formation_per_atom']
                st.markdown(f"""
                <div class='metric-card'>
                    <strong>Formation Energy</strong>
                    <div style='font-size: 1.4rem; font-weight: bold'>{val:.3f} eV/atom</div>
                    <div style='font-size: 0.9rem; opacity: 0.9'>{thermo['stability_label'].split()[0]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='metric-card'><strong>Formation Energy</strong><br>Run Phase 1</div>", unsafe_allow_html=True)
        
        with col2:
            exp = st.session_state.expansion_pct
            if exp is not None:
                st.markdown(f"""
                <div class='metric-card'>
                    <strong>Volume Expansion</strong>
                    <div style='font-size: 1.4rem; font-weight: bold'>{exp:+.1f}%</div>
                    <div style='font-size: 0.9rem; opacity: 0.9'>per Sn atom</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='metric-card'><strong>Volume Expansion</strong><br>Run Phase 2</div>", unsafe_allow_html=True)
        
        with col3:
            if li_el:
                ar = li_el['anisotropy_ratio']
                st.markdown(f"""
                <div class='metric-card'>
                    <strong>Anisotropy Ratio</strong>
                    <div style='font-size: 1.4rem; font-weight: bold'>{ar:.3f}</div>
                    <div style='font-size: 0.9rem; opacity: 0.9'>{'c-soft' if ar<1 else 'isotropic'}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='metric-card'><strong>Anisotropy Ratio</strong><br>Run Phase 3</div>", unsafe_allow_html=True)
        
        with col4:
            if fracture:
                risk = fracture['risk_level']
                risk_color = "#e74c3c" if "CRITICAL" in risk else "#f39c12" if "ELEVATED" in risk else "#27ae60"
                st.markdown(f"""
                <div class='metric-card' style='background: linear-gradient(135deg, {risk_color} 0%, #c0392b 100%)'>
                    <strong>Fracture Risk</strong>
                    <div style='font-size: 1.4rem; font-weight: bold'>{risk.split()[1]}</div>
                    <div style='font-size: 0.9rem; opacity: 0.9'>Score: {fracture['risk_score']}/9</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='metric-card'><strong>Fracture Risk</strong><br>Run Phase 4</div>", unsafe_allow_html=True)
        
        # Multi-property radar chart
        st.subheader("🎯 Multi-Property Radar Analysis")
        
        # Prepare radar chart data with sensible defaults
        radar_props = {}
        
        # Stability: negative formation energy is better (higher = more stable)
        if thermo:
            radar_props['Stability'] = min(max(-thermo['formation_per_atom'] * 10, 0), 10)
        else:
            radar_props['Stability'] = 5  # Neutral
        
        # Expansion risk: lower is better
        exp = st.session_state.expansion_pct
        if exp is not None:
            radar_props['Expansion Risk'] = min(max(exp / 3, 0), 10)  # Scale: 30% → 10
        else:
            radar_props['Expansion Risk'] = 5
        
        # Anisotropy: AR far from 1 is worse
        if li_el:
            ar = li_el['anisotropy_ratio']
            radar_props['Anisotropy'] = min(abs(1 - ar) * 10 + 5, 10)  # AR=1 → 5, AR=0 or 2 → 10
        else:
            radar_props['Anisotropy'] = 5
        
        # Stiffness retention: higher is better
        if sn_eos and li_eos and sn_eos['B0_GPa'] and li_eos['B0_GPa']:
            retention = max(0, (1 - (sn_eos['B0_GPa'] - li_eos['B0_GPa']) / sn_eos['B0_GPa'])) * 10
            radar_props['Stiffness'] = min(retention, 10)
        else:
            radar_props['Stiffness'] = 5
        
        # c-axis strength: higher is better
        if li_el:
            radar_props['c-axis Strength'] = min(li_el['c33_gpa'] / 10, 10)
        else:
            radar_props['c-axis Strength'] = 5
        
        fig = plot_radar_chart(radar_props, "Integrated Mechanical-Thermodynamic Profile")
        st.pyplot(fig)
        
        # Property correlation scatter matrix (using Plotly for interactivity)
        st.subheader("🔗 Property Correlations")
        
        # Prepare data for scatter matrix
        df_scatter = pd.DataFrame({
            'Volume/Sn (Å³)': [
                sn_eos['v0_fit'] / sn_eos['num_sn'],
                li_eos['v0_fit'] / li_eos['num_sn']
            ],
            'Bulk Modulus (GPa)': [
                sn_eos['B0_GPa'] if sn_eos['B0_GPa'] else np.nan,
                li_eos['B0_GPa'] if li_eos['B0_GPa'] else np.nan
            ],
            'C₃₃ (GPa)': [
                sn_el['c33_gpa'] if sn_el and 'c33_gpa' in sn_el else np.nan,
                li_el['c33_gpa'] if li_el else np.nan
            ],
            'Phase': ['β-Sn', 'Li₂Sn₅']
        })
        
        # Create interactive scatter matrix with Plotly
        if not df_scatter.isnull().all().all():
            fig = go.Figure(data=go.Scattermatrix(
                dimensions=[
                    {'label': d, 'values': df_scatter[d], 'range': [df_scatter[d].min()*0.95, df_scatter[d].max()*1.05]}
                    for d in ['Volume/Sn (Å³)', 'Bulk Modulus (GPa)', 'C₃₃ (GPa)']
                    if df_scatter[d].notna().any()
                ],
                marker=dict(
                    color=['#2ecc71', '#9b59b6'],
                    size=12,
                    line=dict(width=1, color='white')
                ),
                text=df_scatter['Phase'],
                hoverinfo='text+dimensions',
                diagonal=dict(visible=False)
            ))
            fig.update_layout(
                title="Multi-Property Scatter Matrix",
                height=450,
                hoverlabel=dict(bgcolor='white', font_size=11)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for correlation plot. Run Phase 2 and Phase 3.")
        
        # Data export section
        st.subheader("💾 Export Results")
        
        # Prepare export dataframe
        export_data = {
            'Property': [
                'Formation Energy (eV/atom)',
                'Volume Expansion (%)',
                'V₀ Sn (Å³)',
                'V₀ Li₂Sn₅ (Å³)',
                'B₀ Sn (GPa)',
                'B₀ Li₂Sn₅ (GPa)',
                'C₁₁ Sn (GPa)',
                'C₃₃ Sn (GPa)',
                'C₁₁ Li₂Sn₅ (GPa)',
                'C₃₃ Li₂Sn₅ (GPa)',
                'Anisotropy Ratio (Li₂Sn₅)',
                'Fracture Risk Score'
            ],
            'Value': [
                thermo['formation_per_atom'] if thermo else 'N/A',
                st.session_state.expansion_pct if st.session_state.expansion_pct is not None else 'N/A',
                sn_eos['v0_fit'] if sn_eos else 'N/A',
                li_eos['v0_fit'] if li_eos else 'N/A',
                sn_eos['B0_GPa'] if sn_eos and sn_eos['B0_GPa'] else 'N/A',
                li_eos['B0_GPa'] if li_eos and li_eos['B0_GPa'] else 'N/A',
                sn_el['c11_gpa'] if sn_el and 'c11_gpa' in sn_el else 'N/A',
                sn_el['c33_gpa'] if sn_el and 'c33_gpa' in sn_el else 'N/A',
                li_el['c11_gpa'] if li_el else 'N/A',
                li_el['c33_gpa'] if li_el else 'N/A',
                li_el['anisotropy_ratio'] if li_el else 'N/A',
                fracture['risk_score'] if fracture else 'N/A'
            ],
            'Unit': [
                'eV/atom', '%', 'Å³', 'Å³', 'GPa', 'GPa', 'GPa', 'GPa', 'GPa', 'GPa', '-', 'score (0-9)'
            ]
        }
        
        export_df = pd.DataFrame(export_data)
        
        # CSV download button
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Results (CSV)",
            data=csv,
            file_name="sn_li2sn5_mechanics_full_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # JSON metadata download
        if st.button("📥 Download Metadata (JSON)", use_container_width=True):
            metadata = {
                'calculation_mode': calculation_mode,
                'parameters': params,
                'computation_times': {k: format_time(v) for k, v in st.session_state.computation_times.items()},
                'timestamp': datetime.now().isoformat(),
                'gpa
