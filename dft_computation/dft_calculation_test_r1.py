#!/usr/bin/env python3
"""
GPAW + ASE Integration Test - FIXED v2.2.0
====================================
Complete fix for calculator attachment and method detection issues

VERSION 2.2.0 - CRITICAL FIXES:
1. ✅ FIXED Calculator attachment persistence in session state
2. ✅ FIXED Method detection - Proper inheritance from ASE Calculator
3. ✅ FIXED atoms.calc reference - Now properly stored and retrieved
4. ✅ FIXED Session state handling for calculator objects
5. ✅ Added atoms.copy() fix to preserve calculator attachment
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS
from ase.calculators.calculator import Calculator, all_changes
import sys
import traceback
from datetime import datetime
import os
import warnings

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="GPAW+ASE Integration Test",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 GPAW + ASE Integration Test")
st.markdown("**Complete fix for calculator attachment and method detection**")

# ============================================================================
# GPAW IMPORT WITH ERROR HANDLING
# ============================================================================
try:
    from gpaw import GPAW, PW
    from ase.calculators.calculator import Calculator as ASECalculator
    GPAW_AVAILABLE = True
    try:
        import gpaw
        GPAW_VERSION = gpaw.__version__
    except:
        GPAW_VERSION = "Unknown"
    st.success(f"✅ GPAW is installed (Version: {GPAW_VERSION})")
except ImportError as e:
    GPAW_AVAILABLE = False
    GPAW_VERSION = None
    st.error(f"❌ GPAW is NOT installed: {e}")
    st.info("Install with: `pip install gpaw` or `conda install -c conda-forge gpaw`")

# ============================================================================
# 🔧🔧🔧 FIXED v2.2.0: DummyCalculator with proper ASE inheritance
# ============================================================================
class DummyCalculator(Calculator):
    """
    Dummy calculator that properly implements ASE Calculator interface.
    CRITICAL: Must inherit from ASE Calculator base class.
    """
    
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']
    
    def __init__(self, atoms=None, ecut=350, xc='PBE', kpts=(4,4,4), **kwargs):
        """Initialize DummyCalculator with all required attributes."""
        # 🔧 CRITICAL: Call parent Calculator __init__ first
        super().__init__(**kwargs)
        
        # Store all parameters as instance attributes
        self.atoms = atoms
        self.ecut = ecut
        self.xc = xc
        self.kpts = kpts
        
        # 🔧 CRITICAL: Initialize results dictionary
        self.results = {}
        
    def calculate(self, atoms=None, properties=['energy'], 
                  system_changes=all_changes):
        """
        REQUIRED method for ASE Calculator interface.
        This is called by ASE whenever energy/forces/stress are needed.
        """
        # 🔧 CRITICAL: Call parent calculate first (handles atoms attachment)
        super().calculate(atoms, properties, system_changes)
        
        # If atoms provided, update internal reference
        if atoms is not None:
            self.atoms = atoms.copy()
        
        # Ensure we have atoms to work with
        if self.atoms is None:
            raise RuntimeError("DummyCalculator has no atoms attached")
        
        # Get atom counts
        n_atoms = len(self.atoms)
        symbols = self.atoms.get_chemical_symbols()
        n_sn = sum(1 for s in symbols if 'Sn' in s)
        n_li = sum(1 for s in symbols if 'Li' in s)
        
        # Reference energies (PBE-typical values)
        e_sn_ref = -3.152
        e_li_ref = -1.908
        
        # Compute energy with simple volume penalty term
        if hasattr(self.atoms, 'get_volume'):
            vol = self.atoms.get_volume()
            vol_term = 0.001 * (vol - 100)**2 / 100
        else:
            vol_term = 0.0
            
        energy = n_sn * e_sn_ref + n_li * e_li_ref + vol_term
        
        # 🔧 CRITICAL: Store results in self.results with EXACT keys ASE expects
        self.results['energy'] = float(energy)
        self.results['free_energy'] = float(energy)
        
        # Forces: zero array with proper shape (N,3) and dtype=float64
        self.results['forces'] = np.zeros((n_atoms, 3), dtype=np.float64)
        
        # Stress: 6-element Voigt array with dtype=float64
        self.results['stress'] = np.zeros(6, dtype=np.float64)
        
        return self.results
    
    # 🔧 CRITICAL: Implement get methods that ASE calls
    def get_potential_energy(self, atoms=None):
        """Return potential energy - called by atoms.get_potential_energy()"""
        if atoms is not None:
            self.atoms = atoms
        if 'energy' not in self.results or self.atoms is None:
            self.calculate(self.atoms)
        return self.results.get('energy', 0.0)
    
    def get_forces(self, atoms=None):
        """Return forces - called by atoms.get_forces()"""
        if atoms is not None:
            self.atoms = atoms
        if 'forces' not in self.results or self.atoms is None:
            self.calculate(self.atoms)
        return self.results.get('forces', np.zeros((len(self.atoms or []), 3), dtype=np.float64))
    
    def get_stress(self, atoms=None):
        """Return stress - called by atoms.get_stress()"""
        if atoms is not None:
            self.atoms = atoms
        if 'stress' not in self.results or self.atoms is None:
            self.calculate(self.atoms)
        return self.results.get('stress', np.zeros(6, dtype=np.float64))

# ============================================================================
# GPAW Stub Class (for when GPAW is not available)
# ============================================================================
class GPAW_Stub:
    """Stub GPAW calculator for testing when real GPAW is not available."""
    
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
        self.results = {}
        
    def set(self, **kwargs):
        """Set calculator parameters"""
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def attach_atoms(self, atoms):
        """Attach calculator to atoms"""
        self.atoms = atoms
        atoms.calc = self
    
    def get_potential_energy(self, atoms=None):
        """Return dummy energy"""
        if atoms is not None:
            self.atoms = atoms
        if self.atoms is not None:
            return -len(self.atoms) * 3.0
        return 0.0
    
    def get_forces(self, atoms=None):
        """Return zero forces with proper dtype"""
        if atoms is not None:
            self.atoms = atoms
        if self.atoms is not None:
            return np.zeros((len(self.atoms), 3), dtype=np.float64)
        return np.array([], dtype=np.float64).reshape(0, 3)
    
    def get_stress(self, atoms=None):
        """Return zero stress with proper dtype"""
        return np.zeros(6, dtype=np.float64)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def log_message(message, level="info"):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if level == "error":
        st.error(f"[{timestamp}] {message}")
    elif level == "warning":
        st.warning(f"[{timestamp}] {message}")
    elif level == "success":
        st.success(f"[{timestamp}] {message}")
    else:
        st.info(f"[{timestamp}] {message}")

def format_time(seconds):
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        return f"{seconds/3600:.2f} hours"

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'test_atoms': None,
        'calc_attached': False,
        'calc': None,
        'enable_detailed_logging': False,
        'last_error': None,
        'SAFE_FORCE_REAL_DFT': True,
        'calc_params': None,  # Store calculator parameters instead of object
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# SIDEBAR SETTINGS
# ============================================================================
st.sidebar.header("⚙️ Test Settings")

st.session_state['SAFE_FORCE_REAL_DFT'] = st.sidebar.checkbox(
    "🛡️ Safe Mode: Use Dummy Calculator",
    value=not GPAW_AVAILABLE,
    help="When enabled, always use DummyCalculator to prevent crashes."
)

st.session_state['enable_detailed_logging'] = st.sidebar.checkbox(
    "Enable detailed logging",
    value=False,
    help="Show detailed calculation logs for debugging"
)

if st.sidebar.button("🔄 Reset Session", use_container_width=True):
    for key in ['test_atoms', 'calc_attached', 'calc', 'last_error', 'calc_params']:
        if key in st.session_state:
            st.session_state[key] = None if key not in ['enable_detailed_logging', 'SAFE_FORCE_REAL_DFT'] else False
    st.rerun()

# ============================================================================
# TEST STRUCTURE CREATION
# ============================================================================
st.header("1️⃣ Test Structure Creation")

structure_type = st.selectbox(
    "Select test structure",
    ["Bulk Sn (BCT)", "Bulk Li (BCC)", "Simple H₂ molecule"]
)

atoms = None
if structure_type == "Bulk Sn (BCT)":
    with st.spinner("Creating β-Sn structure..."):
        atoms = bulk('Sn', 'bct', a=5.83, c=3.18)
        st.success(f"✅ Created β-Sn: {len(atoms)} atoms, V = {atoms.get_volume():.2f} Å³")
        with st.expander("View structure details"):
            st.code(f"Cell:\n{atoms.get_cell().array}")
            st.code(f"Positions:\n{atoms.get_positions()}")
            
elif structure_type == "Bulk Li (BCC)":
    with st.spinner("Creating Li structure..."):
        atoms = bulk('Li', 'bcc', a=3.51)
        st.success(f"✅ Created Li: {len(atoms)} atoms, V = {atoms.get_volume():.2f} Å³")
        with st.expander("View structure details"):
            st.code(f"Cell:\n{atoms.get_cell().array}")
            
elif structure_type == "Simple H₂ molecule":
    with st.spinner("Creating H₂ molecule..."):
        atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.74)])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(False)
        st.success(f"✅ Created H₂: {len(atoms)} atoms")
        with st.expander("View structure details"):
            st.code(f"Positions:\n{atoms.get_positions()}")

# Store atoms in session state for later use
if atoms is not None:
    st.session_state['test_atoms'] = atoms

# ============================================================================
# CALCULATOR SETUP
# ============================================================================
st.header("2️⃣ Calculator Setup")

col1, col2 = st.columns(2)
with col1:
    ecut = st.slider("Plane-wave cutoff (eV)", 100, 600, 350, 50)
    kpts = st.selectbox("k-point grid", [(2,2,2), (3,3,3), (4,4,4), (6,6,6)], index=1)
with col2:
    xc = st.selectbox("Exchange-correlation functional", ["PBE", "LDA", "BLYP"], index=0)
    fmax = st.slider("Force convergence (eV/Å)", 0.001, 0.1, 0.05, 0.005)

st.markdown("### Calculator Parameters:")
st.json({
    "mode": f"PW(ecut={ecut} eV)",
    "xc": xc,
    "kpts": kpts,
    "convergence": {"energy": 1e-5, "density": 1e-4},
    "fmax": f"{fmax} eV/Å"
})

# ============================================================================
# CALCULATOR ATTACHMENT TEST - FIXED v2.2.0
# ============================================================================
st.header("3️⃣ Calculator Attachment Test")

if st.button("🔧 Attach Calculator to Atoms", use_container_width=True):
    try:
        atoms = st.session_state.get('test_atoms')
        if atoms is None:
            st.error("❌ No atoms created yet. Please create a structure first.")
            st.stop()
        
        # 🔧🔧🔧 FIXED: Create fresh calculator instance each time
        if not GPAW_AVAILABLE or st.session_state['SAFE_FORCE_REAL_DFT']:
            if not GPAW_AVAILABLE:
                st.warning("⚠️ GPAW not available - using DummyCalculator")
            else:
                st.info("ℹ️ Safe Mode enabled - using DummyCalculator")
            
            # Create fresh calculator
            calc = DummyCalculator(
                atoms=atoms,
                ecut=ecut,
                xc=xc,
                kpts=kpts
            )
        else:
            # Use real GPAW
            calc = GPAW(
                mode=PW(ecut),
                xc=xc,
                kpts=kpts,
                txt=None,
                convergence={'energy': 1e-5, 'density': 1e-4},
                maxiter=50,
                occupations={'name': 'fermi-dirac', 'width': 0.1}
            )
        
        # 🔧🔧🔧 CRITICAL: Attach calculator to atoms
        atoms.calc = calc
        
        # 🔧🔧🔧 CRITICAL: Store both atoms and calc in session state
        st.session_state['test_atoms'] = atoms
        st.session_state['calc'] = calc
        st.session_state['calc_attached'] = True
        st.session_state['calc_params'] = {
            'ecut': ecut,
            'xc': xc,
            'kpts': kpts,
            'type': 'DummyCalculator' if (not GPAW_AVAILABLE or st.session_state['SAFE_FORCE_REAL_DFT']) else 'GPAW'
        }
        
        st.success("✅ Calculator successfully attached to Atoms object!")
        st.info(f"Calculator type: {type(atoms.calc).__name__}")
        
        # Verify calculator attachment
        st.markdown("### Verification:")
        
        # Check if atoms has calculator
        has_calc = hasattr(atoms, 'calc') and atoms.calc is not None
        st.write(f"atoms.calc attached: {has_calc}")
        
        if has_calc:
            # Check for required methods
            methods = ['get_potential_energy', 'get_forces', 'get_stress', 'calculate']
            for method in methods:
                has_method = hasattr(atoms.calc, method)
                if has_method:
                    st.success(f"✅ {method}: Available")
                else:
                    st.error(f"❌ {method}: Missing")
        
    except Exception as e:
        st.error(f"❌ Failed to attach calculator: {e}")
        st.exception(e)
        st.session_state['calc_attached'] = False
        st.session_state['last_error'] = str(e)

# ============================================================================
# ENERGY & FORCES CALCULATION TEST - FIXED v2.2.0
# ============================================================================
st.header("4️⃣ Energy & Forces Calculation Test")

if st.session_state.get('calc_attached', False):
    if st.button("⚡ Calculate Energy & Forces", use_container_width=True):
        try:
            # 🔧🔧🔧 FIXED: Get atoms from session state (which has calc attached)
            atoms = st.session_state.get('test_atoms')
            
            if atoms is None:
                st.error("❌ No atoms in session state")
                st.stop()
            
            # 🔧🔧🔧 CRITICAL: Check if calculator is still attached
            if not hasattr(atoms, 'calc') or atoms.calc is None:
                st.error("❌ Calculator not attached to atoms!")
                st.info("Please re-run Step 3: Attach Calculator")
                st.stop()
            
            with st.spinner("Running calculation..."):
                # Test 1: Calculate energy
                st.markdown("### Energy Calculation:")
                energy = atoms.get_potential_energy()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Energy", f"{energy:.6f} eV")
                with col2:
                    st.metric("Energy per Atom", f"{energy/len(atoms):.6f} eV/atom")
                
                # Test 2: Calculate forces
                st.markdown("### Forces:")
                forces = atoms.get_forces()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Max Force", f"{np.abs(forces).max():.6f} eV/Å")
                with col2:
                    st.metric("Avg Force", f"{np.abs(forces).mean():.6f} eV/Å")
                
                # Show forces dataframe
                if len(forces) > 0:
                    df_forces = pd.DataFrame(forces, columns=['Fx (eV/Å)', 'Fy (eV/Å)', 'Fz (eV/Å)'])
                    df_forces.index = [f"Atom {i}" for i in range(len(forces))]
                    st.dataframe(df_forces, use_container_width=True)
                
                # Test 3: Calculate stress (if periodic)
                if atoms.pbc.any():
                    st.markdown("### Stress Tensor:")
                    stress = atoms.get_stress()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.code(f"Stress (eV/Å³):\n{stress}")
                    with col2:
                        stress_gpa = stress * 160.217
                        st.code(f"Stress (GPa):\n{stress_gpa}")
                
                st.success("✅ All calculations completed successfully!")
                
        except RuntimeError as e:
            if "no calculator" in str(e).lower():
                st.error(f"❌ RuntimeError: {e}")
                st.markdown("""
                **This error means the calculator is not attached!**
                
                The atoms object lost its calculator reference. This can happen when:
                1. atoms.copy() was called without preserving calc
                2. Session state was reset
                3. Calculator object was not serializable
                
                **Solution**: Re-run Step 3 to re-attach the calculator.
                """)
            else:
                st.error(f"❌ RuntimeError: {e}")
                st.exception(e)
            st.session_state['last_error'] = str(e)
        except Exception as e:
            st.error(f"❌ Calculation failed: {e}")
            st.exception(e)
            st.session_state['last_error'] = str(e)
else:
    st.warning("⚠️ Please attach calculator first (Step 3)")

# ============================================================================
# BFGS OPTIMIZATION TEST - FIXED v2.2.0
# ============================================================================
st.header("5️⃣ BFGS Optimization Test")
st.markdown("*Complete implementation with proper calculator handling*")

if st.session_state.get('calc_attached', False):
    col1, col2 = st.columns(2)
    with col1:
        max_steps = st.slider("Max optimization steps", 10, 200, 50, 10)
    with col2:
        fmax_opt = st.slider("Force tolerance (eV/Å)", 0.001, 0.1, 0.05, 0.005, key="fmax_opt")
    
    if st.button("🚀 Run BFGS Optimization", use_container_width=True, key="btn_bfgs"):
        try:
            # 🔧🔧🔧 FIXED: Get atoms from session state
            atoms = st.session_state.get('test_atoms')
            
            if atoms is None:
                st.error("❌ No atoms in session state")
                st.stop()
            
            # 🔧🔧🔧 CRITICAL: Check calculator attachment
            if not hasattr(atoms, 'calc') or atoms.calc is None:
                st.error("❌ Calculator not attached!")
                st.info("Please re-run Step 3: Attach Calculator")
                st.stop()
            
            # 🔧🔧🔧 FIXED: Create a fresh copy with calculator preserved
            # Use atoms.copy() which should preserve calc, but verify
            atoms_opt = atoms.copy()
            if atoms_opt.calc is None and atoms.calc is not None:
                atoms_opt.calc = atoms.calc
            
            with st.spinner(f"Running BFGS optimization..."):
                start_time = datetime.now()
                
                # Run BFGS optimization
                opt = BFGS(atoms_opt, logfile=None)
                converged = opt.run(fmax=fmax_opt, steps=max_steps)
                
                elapsed = (datetime.now() - start_time).total_seconds()
                
                if converged:
                    st.success(f"✅ Optimization converged in {format_time(elapsed)}!")
                else:
                    st.warning(f"⚠️ Optimization reached max steps in {format_time(elapsed)}")
                
                # Show results
                st.markdown("### Optimization Results:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Energy", f"{atoms_opt.get_potential_energy():.6f} eV")
                with col2:
                    forces = atoms_opt.get_forces()
                    st.metric("Max Final Force", f"{np.abs(forces).max():.6f} eV/Å")
                with col3:
                    st.metric("Cell Volume", f"{atoms_opt.get_volume():.2f} Å³")
                
                # Show atomic positions
                with st.expander("View final atomic positions"):
                    st.code(f"Positions:\n{atoms_opt.get_positions()}")
                
                # Update session state with optimized structure
                st.session_state['test_atoms'] = atoms_opt
                
        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")
            st.exception(e)
            st.session_state['last_error'] = str(e)
else:
    st.warning("⚠️ Please attach calculator first (Step 3)")

# ============================================================================
# DIAGNOSTIC INFORMATION
# ============================================================================
st.header("📊 System Information")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("GPAW Available", "✅ Yes" if GPAW_AVAILABLE else "❌ No")
    st.metric("GPAW Version", GPAW_VERSION or "N/A")
with col2:
    st.metric("Python Version", sys.version.split()[0])
    st.metric("Platform", sys.platform)
with col3:
    import multiprocessing as mp
    st.metric("CPU Cores", mp.cpu_count())
    try:
        import ase
        st.metric("ASE Version", ase.__version__)
    except:
        st.metric("ASE Version", "Unknown")

# Display session state info
with st.expander("📋 Session State Debug"):
    st.write("calc_attached:", st.session_state.get('calc_attached', False))
    st.write("test_atoms exists:", st.session_state.get('test_atoms') is not None)
    if st.session_state.get('test_atoms') is not None:
        atoms = st.session_state['test_atoms']
        st.write("atoms has calc:", hasattr(atoms, 'calc') and atoms.calc is not None)
        if hasattr(atoms, 'calc') and atoms.calc is not None:
            st.write("calc type:", type(atoms.calc).__name__)
    st.write("calc_params:", st.session_state.get('calc_params'))

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.85rem; padding: 1rem 0;'>
<strong>GPAW + ASE Integration Test</strong><br>
Complete fix for calculator attachment and method detection issues<br>
<em>Version 2.2.0 - Fixed calculator persistence and method availability</em><br>
Current Mode: {"🛡️ Safe Mode (Dummy Calculator)" if st.session_state['SAFE_FORCE_REAL_DFT'] or not GPAW_AVAILABLE else "🚀 Real GPAW Mode"}
</div>
""", unsafe_allow_html=True)
