#!/usr/bin/env python3
"""
GPAW + ASE Integration Test - FIXED v2.1.1
====================================
Simple Streamlit app to verify GPAW calculator is properly accessed by ASE
Run with: streamlit run dft_calculation_test.py

VERSION 2.1.1 - CRITICAL FIX:
1. ✅ FIXED calculator persistence: store atoms with calculator attached in session state
2. ✅ DummyCalculator properly inherits from ASE Calculator base class
3. ✅ All ASE interface methods implemented for BFGS compatibility
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
st.markdown("**This app verifies GPAW calculator is properly accessed by ASE**")

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
# 🔧 FIXED: DummyCalculator with proper ASE Calculator inheritance
# ============================================================================
class DummyCalculator(Calculator):
    """
    Dummy calculator that properly implements ASE Calculator interface.
    CRITICAL: Must inherit from ase.calculators.calculator.Calculator
    """
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']
    
    def __init__(self, atoms=None, ecut=350, xc='PBE', kpts=(4,4,4), **kwargs):
        # CRITICAL: Call parent Calculator __init__ first
        super().__init__(**kwargs)
        self.atoms = atoms
        self.results = {}
        self.ecut = ecut
        self.xc = xc
        self.kpts = kpts
        
    def calculate(self, atoms=None, properties=['energy'], 
                  system_changes=all_changes):
        """
        REQUIRED method for ASE Calculator interface.
        Called by ASE whenever energy/forces/stress are needed.
        """
        # CRITICAL: Call parent calculate first (handles atoms attachment)
        super().calculate(atoms, properties, system_changes)
        
        if atoms is not None:
            self.atoms = atoms.copy()
        if self.atoms is None:
            raise RuntimeError("DummyCalculator has no atoms attached")
        
        # Simple empirical energy model for testing
        n_atoms = len(self.atoms)
        symbols = self.atoms.get_chemical_symbols()
        n_sn = sum(1 for s in symbols if 'Sn' in s)
        n_li = sum(1 for s in symbols if 'Li' in s)
        
        e_sn_ref = -3.152  # PBE-typical reference energy per Sn atom
        e_li_ref = -1.908  # PBE-typical reference energy per Li atom
        
        # Volume penalty term for testing structural changes
        if hasattr(self.atoms, 'get_volume'):
            vol = self.atoms.get_volume()
            vol_term = 0.001 * (vol - 100)**2 / 100
        else:
            vol_term = 0.0
            
        energy = n_sn * e_sn_ref + n_li * e_li_ref + vol_term
        
        # CRITICAL: Store results with EXACT keys ASE expects
        self.results['energy'] = float(energy)
        self.results['free_energy'] = float(energy)
        self.results['forces'] = np.zeros((n_atoms, 3), dtype=np.float64)
        self.results['stress'] = np.zeros(6, dtype=np.float64)
        
        # Optional logging
        if st.session_state.get('enable_detailed_logging', False):
            st.info(f"🧮 DummyCalculator: E={energy:.3f} eV for {n_sn}Sn+{n_li}Li")
        
        return self.results

# ============================================================================
# GPAW Stub Class (fallback when GPAW not available)
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
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def attach_atoms(self, atoms):
        self.atoms = atoms
        atoms.calc = self
    
    def get_potential_energy(self):
        if self.atoms is not None:
            return -len(self.atoms) * 3.0
        return 0.0
    
    def get_forces(self):
        if self.atoms is not None:
            return np.zeros((len(self.atoms), 3), dtype=np.float64)
        return np.array([], dtype=np.float64).reshape(0, 3)
    
    def get_stress(self):
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
    help="When enabled, always use DummyCalculator to prevent crashes on Streamlit Cloud. Disable only if you have real GPAW installed locally."
)

st.session_state['enable_detailed_logging'] = st.sidebar.checkbox(
    "Enable detailed logging",
    value=False,
    help="Show detailed calculation logs for debugging"
)

if st.sidebar.button("🔄 Reset Session", use_container_width=True):
    for key in ['test_atoms', 'calc_attached', 'calc', 'last_error']:
        if key in st.session_state:
            st.session_state[key] = None if key != 'enable_detailed_logging' else False
    st.rerun()

# ============================================================================
# 1️⃣ TEST STRUCTURE CREATION
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
# 2️⃣ CALCULATOR SETUP
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
# 3️⃣ CALCULATOR ATTACHMENT TEST - 🔧 CRITICAL FIX HERE
# ============================================================================
st.header("3️⃣ Calculator Attachment Test")

if st.button("🔧 Attach Calculator to Atoms", use_container_width=True):
    try:
        atoms = st.session_state.get('test_atoms')
        if atoms is None:
            st.error("❌ No atoms created yet. Please create a structure first.")
            st.stop()
        
        # Use DummyCalculator if SAFE_FORCE_REAL_DFT is enabled or GPAW unavailable
        if not GPAW_AVAILABLE or st.session_state['SAFE_FORCE_REAL_DFT']:
            if not GPAW_AVAILABLE:
                st.warning("⚠️ GPAW not available - using DummyCalculator")
            else:
                st.info("ℹ️ Safe Mode enabled - using DummyCalculator")
            calc = DummyCalculator(
                atoms=atoms,
                ecut=ecut,
                xc=xc,
                kpts=kpts
            )
        else:
            # Use real GPAW (only if available and Safe Mode disabled)
            calc = GPAW(
                mode=PW(ecut),
                xc=xc,
                kpts=kpts,
                txt=None,
                convergence={'energy': 1e-5, 'density': 1e-4},
                maxiter=50,
                occupations={'name': 'fermi-dirac', 'width': 0.1}
            )
        
        # 🔧 CRITICAL: Attach calculator to atoms
        atoms.calc = calc
        
        # 🔧🔧🔧 FIX: Update session state with the atoms object that now has calculator attached
        # This is the critical fix - without this, later steps retrieve calculator-less atoms
        st.session_state['test_atoms'] = atoms
        
        st.success("✅ Calculator successfully attached to Atoms object!")
        st.info(f"Calculator type: {type(atoms.calc).__name__}")
        
        # Verify calculator properties with safe attribute checking
        st.markdown("### Calculator Properties:")
        calc_properties = {}
        calc_properties['has_atoms'] = hasattr(atoms.calc, 'atoms') and atoms.calc.atoms is not None
        
        if hasattr(atoms.calc, 'xc'):
            calc_properties['xc_functional'] = atoms.calc.xc
        else:
            calc_properties['xc_functional'] = 'N/A'
        
        if hasattr(atoms.calc, 'kpts'):
            calc_properties['kpts'] = str(atoms.calc.kpts)
        else:
            calc_properties['kpts'] = 'N/A'
        
        if hasattr(atoms.calc, 'ecut'):
            calc_properties['ecut'] = f"{atoms.calc.ecut} eV"
        elif hasattr(atoms.calc, 'mode') and hasattr(atoms.calc.mode, 'ecut'):
            calc_properties['ecut'] = f"{atoms.calc.mode.ecut} eV"
        else:
            calc_properties['ecut'] = 'N/A'
        
        st.json(calc_properties)
        
        # Save calculator state to session
        st.session_state['calc_attached'] = True
        st.session_state['calc'] = calc
        
    except AttributeError as e:
        st.error(f"❌ AttributeError: {e}")
        st.markdown("**This indicates the calculator is not properly initialized**")
        st.info("Common causes:")
        st.markdown("""
        1. Calculator missing required attributes (xc, kpts, etc.)
        2. Calculator not fully initialized before attachment
        3. Attribute accessed before being set
        """)
        st.exception(e)
        st.session_state['last_error'] = str(e)
    except Exception as e:
        st.error(f"❌ Failed to attach calculator: {e}")
        st.exception(e)
        st.session_state['calc_attached'] = False
        st.session_state['last_error'] = str(e)

# ============================================================================
# 4️⃣ ENERGY & FORCES CALCULATION TEST
# ============================================================================
st.header("4️⃣ Energy & Forces Calculation Test")

if st.session_state.get('calc_attached', False):
    if st.button("⚡ Calculate Energy & Forces", use_container_width=True):
        try:
            atoms = st.session_state['test_atoms']
            
            with st.spinner("Running calculation..."):
                # Test 1: Check if calculator has required methods
                st.markdown("### Method Availability Check:")
                methods_to_check = ['get_potential_energy', 'get_forces', 'get_stress', 'calculate']
                for method in methods_to_check:
                    has_method = hasattr(atoms.calc, method)
                    if has_method:
                        st.success(f"✅ {method}: Available")
                    else:
                        st.error(f"❌ {method}: Missing")
                
                # Test 2: Calculate energy
                st.markdown("### Energy Calculation:")
                energy = atoms.get_potential_energy()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Energy", f"{energy:.6f} eV")
                with col2:
                    st.metric("Energy per Atom", f"{energy/len(atoms):.6f} eV/atom")
                
                # Test 3: Calculate forces
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
                
                # Test 4: Calculate stress (if periodic)
                if atoms.pbc.any():
                    st.markdown("### Stress Tensor:")
                    stress = atoms.get_stress()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.code(f"Stress (eV/Å³):\n{stress}")
                    with col2:
                        stress_gpa = stress * 160.217  # Convert to GPa
                        st.code(f"Stress (GPa):\n{stress_gpa}")
                
                st.success("✅ All calculator methods working correctly!")
                
        except AttributeError as e:
            st.error(f"❌ AttributeError: {e}")
            st.markdown("**This indicates the calculator is not properly attached**")
            st.info("Try: `atoms.calc = calc` before calling `atoms.get_potential_energy()`")
            st.exception(e)
            st.session_state['last_error'] = str(e)
        except Exception as e:
            st.error(f"❌ Calculation failed: {e}")
            st.exception(e)
            st.session_state['last_error'] = str(e)
else:
    st.warning("⚠️ Please attach calculator first (Step 3)")

# ============================================================================
# 5️⃣ BFGS OPTIMIZATION TEST (THE CRITICAL TEST)
# ============================================================================
st.header("5️⃣ BFGS Optimization Test")
st.markdown("*This is the test that triggers the original error if DummyCalculator is not properly implemented*")

if st.session_state.get('calc_attached', False):
    col1, col2 = st.columns(2)
    with col1:
        max_steps = st.slider("Max optimization steps", 10, 200, 50, 10)
    with col2:
        fmax_opt = st.slider("Force tolerance (eV/Å)", 0.001, 0.1, 0.05, 0.005)
    
    if st.button("🚀 Run BFGS Optimization", use_container_width=True, key="btn_bfgs"):
        try:
            # Get atoms with calculator attached from session state
            atoms = st.session_state['test_atoms'].copy()
            atoms.calc = st.session_state['calc']
            
            with st.spinner(f"Running BFGS optimization (fmax={fmax_opt} eV/Å, max_steps={max_steps})..."):
                start_time = datetime.now()
                
                # 🔧 This is where the original error occurred
                # With the fixed DummyCalculator, this should now work
                opt = BFGS(atoms, logfile=None)
                converged = opt.run(fmax=fmax_opt, steps=max_steps)
                
                elapsed = (datetime.now() - start_time).total_seconds()
                
                if converged:
                    st.success(f"✅ Optimization converged in {format_time(elapsed)}!")
                else:
                    st.warning(f"⚠️ Optimization did not fully converge (reached max steps) in {format_time(elapsed)}")
                
                # Show results
                st.markdown("### Optimization Results:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Energy", f"{atoms.get_potential_energy():.6f} eV")
                with col2:
                    forces = atoms.get_forces()
                    st.metric("Max Final Force", f"{np.abs(forces).max():.6f} eV/Å")
                with col3:
                    st.metric("Cell Volume", f"{atoms.get_volume():.2f} Å³")
                
                # Show atomic positions
                with st.expander("View final atomic positions"):
                    st.code(f"Positions:\n{atoms.get_positions()}")
                
        except AttributeError as e:
            st.error(f"❌ AttributeError during optimization: {e}")
            st.markdown("""
            **This is the error we're fixing!**
            
            If you see this, the DummyCalculator is not properly implementing
            the ASE Calculator interface. Make sure:
            1. It inherits from `ase.calculators.calculator.Calculator`
            2. It implements the `calculate()` method
            3. It stores results in `self.results` with keys: 'energy', 'forces', 'stress'
            """)
            st.exception(e)
            st.session_state['last_error'] = str(e)
        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")
            st.exception(e)
            st.session_state['last_error'] = str(e)
else:
    st.warning("⚠️ Please attach calculator first (Step 3)")

# ============================================================================
# 📊 SYSTEM INFORMATION
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

# Display detailed version info
with st.expander("📋 Detailed Version Information"):
    st.markdown("### Python Packages:")
    packages = {
        'streamlit': 'streamlit',
        'ase': 'ase',
        'gpaw': 'gpaw',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas'
    }
    for pkg_name, import_name in packages.items():
        try:
            pkg = __import__(import_name)
            version = getattr(pkg, '__version__', 'Unknown')
            st.success(f"✅ {pkg_name}: {version}")
        except ImportError:
            st.error(f"❌ {pkg_name}: Not installed")

# ============================================================================
# 🔧 TROUBLESHOOTING GUIDE
# ============================================================================
st.header("🔧 Troubleshooting Guide")

with st.expander("Common Issues & Solutions"):
    st.markdown("""
    ### Issue: `AttributeError: 'NoneType' object has no attribute 'get_forces'`
    **Cause**: Calculator not attached to atoms before calling `atoms.get_forces()` or optimization
    
    **Solution**:
    ```python
    atoms.calc = calc  # ← Must do this BEFORE optimization or energy/force calls
    opt = BFGS(atoms)
    opt.run(fmax=0.05)
    ```
    
    ### Issue: `AttributeError: 'DummyCalculator' object has no attribute 'xc'`
    **Cause**: DummyCalculator missing required attributes
    
    **Solution**: Ensure all attributes are initialized in `__init__`:
    ```python
    def __init__(self, xc='PBE', ...):
        self.xc = xc  # ← Must store this!
    ```
    
    ### Issue: Optimizer fails with forces error
    **Cause**: DummyCalculator doesn't implement ASE Calculator interface properly
    
    **Solution**: Inherit from ASE Calculator and implement calculate():
    ```python
    from ase.calculators.calculator import Calculator, all_changes
    
    class DummyCalculator(Calculator):
        implemented_properties = ['energy', 'forces', 'stress']
        
        def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            # ... compute energy, forces, stress ...
            self.results['energy'] = energy
            self.results['forces'] = np.zeros((n_atoms, 3), dtype=np.float64)
            self.results['stress'] = np.zeros(6, dtype=np.float64)
    ```
    
    ### Issue: `ImportError: No module named 'gpaw'`
    **Cause**: GPAW not installed
    
    **Solution**:
    ```bash
    pip install gpaw
    # Or with conda:
    conda install -c conda-forge gpaw
    ```
    
    ### Issue: Streamlit Cloud timeout/crash
    **Cause**: Real DFT calculations are too slow for cloud environments
    
    **Solution**: Enable "Safe Mode: Use Dummy Calculator" in sidebar
    """)

# ============================================================================
# 📚 KEY TAKEAWAYS
# ============================================================================
st.header("📚 Key Takeaways")

st.markdown("""
### For the DummyCalculator to work with ASE optimizers:

1. **Inherit from ASE Calculator**: `class DummyCalculator(Calculator)`
2. **Declare implemented properties**: `implemented_properties = ['energy', 'forces', 'stress']`
3. **Implement calculate() method**: This is the entry point ASE uses
4. **Store results in self.results**: Use exact keys: 'energy', 'forces', 'stress', 'free_energy'
5. **Return proper dtypes**: Forces must be `np.ndarray` with `dtype=np.float64`, shape `(N, 3)`
6. **Attach calculator before use**: `atoms.calc = calc` BEFORE calling `atoms.get_forces()` or `opt.run()`
7. **🔧 CRITICAL FIX**: After attaching calculator, update session state: `st.session_state['test_atoms'] = atoms`

### Why the original error occurred:

The ASE BFGS optimizer calls `atoms.get_forces()`, which internally calls `calc.get_forces(atoms)`.
If the calculator doesn't properly implement the ASE interface, this call fails with AttributeError.

By inheriting from `ase.calculators.calculator.Calculator` and implementing `calculate()`,
ASE's base class handles the method dispatch correctly, preventing the error.
""")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.85rem; padding: 1rem 0;'>
<strong>GPAW + ASE Integration Test</strong><br>
Use this app to verify your DFT backend is working before running full analysis<br>
<em>Version 2.1.1 - Fixed calculator persistence in session state</em><br>
Current Mode: {"🛡️ Safe Mode (Dummy Calculator)" if st.session_state['SAFE_FORCE_REAL_DFT'] or not GPAW_AVAILABLE else "🚀 Real GPAW Mode"}
</div>
""", unsafe_allow_html=True)

# Display last error if logging enabled
if st.session_state.get('last_error') and st.session_state.get('enable_detailed_logging', False):
    with st.expander("🐛 Last Error Details (Debug)", expanded=False):
        st.code(st.session_state['last_error'], language="text")
        if st.button("Clear Error", key="clear_err"):
            st.session_state['last_error'] = None
            st.rerun()
