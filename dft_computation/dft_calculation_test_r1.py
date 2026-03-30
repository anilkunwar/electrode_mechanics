#!/usr/bin/env python3
"""
GPAW + ASE Integration Test - FIXED
====================================
Simple Streamlit app to verify GPAW calculator is properly accessed by ASE
Run with: streamlit run dft_calculation_test_r1.py

VERSION 2.0.1 - CRITICAL FIXES:
1. ✅ FIXED GPAW stub class - xc attribute now properly stored
2. ✅ FIXED DummyCalculator - All required attributes initialized
3. ✅ FIXED Calculator property access - Safe attribute checking
4. ✅ Added robust error handling for calculator attachment
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS
import sys
import traceback
from datetime import datetime

# ============================================================================
# GPAW IMPORT WITH ERROR HANDLING
# ============================================================================
st.title("🔬 GPAW + ASE Integration Test")
st.markdown("**This app verifies GPAW calculator is properly accessed by ASE**")

# Check GPAW availability
try:
    from gpaw import GPAW, PW
    from ase.calculators.calculator import Calculator
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
    st.stop()

# ============================================================================
# 🔧🔧🔧 FIXED: GPAW Stub Class with Proper Attribute Storage
# ============================================================================
class GPAW_Stub:
    """
    Stub GPAW calculator for testing when real GPAW is not available.
    🔧 FIXED v2.0.1: All required attributes properly initialized
    """
    
    def __init__(self, mode=None, xc='PBE', kpts=None, txt=None, convergence=None,
                 maxiter=200, occupations=None, **kwargs):
        # 🔧 CRITICAL: Store ALL attributes that might be accessed later
        self.mode = mode
        self.xc = xc  # ← This was missing!
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
    
    def get_potential_energy(self):
        """Return dummy energy"""
        if self.atoms is not None:
            return -len(self.atoms) * 3.0
        return 0.0
    
    def get_forces(self):
        """Return zero forces"""
        if self.atoms is not None:
            return np.zeros((len(self.atoms), 3))
        return np.array([])
    
    def get_stress(self):
        """Return zero stress"""
        return np.zeros(6)

# ============================================================================
# 🔧🔧🔧 FIXED: DummyCalculator with All Required Attributes
# ============================================================================
class DummyCalculator:
    """
    Dummy calculator for demo mode that properly implements ASE Calculator interface.
    🔧 FIXED v2.0.1: All required attributes properly initialized
    """
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, atoms=None, ecut=350, xc='PBE', kpts=(4,4,4), **kwargs):
        # 🔧 CRITICAL: Store ALL attributes that might be accessed
        self.atoms = atoms
        self.results = {}
        self.ecut = ecut
        self.xc = xc  # ← This was missing!
        self.kpts = kpts
        self.kwargs = kwargs
        
    def get_potential_energy(self, force_consistent=False):
        """Return potential energy"""
        if self.atoms is not None:
            n_atoms = len(self.atoms)
            symbols = self.atoms.get_chemical_symbols()
            n_sn = sum(1 for s in symbols if 'Sn' in s)
            n_li = sum(1 for s in symbols if 'Li' in s)
            e_sn_ref = -3.152
            e_li_ref = -1.908
            if hasattr(self.atoms, 'get_volume'):
                vol = self.atoms.get_volume()
                vol_term = 0.001 * (vol - 100)**2 / 100
            else:
                vol_term = 0
            return n_sn * e_sn_ref + n_li * e_li_ref + vol_term
        return -100.0
    
    def get_forces(self, apply_constraint=True):
        """Return forces - MUST return np.ndarray with dtype=float64, shape (N,3)"""
        if self.atoms is not None:
            return np.zeros((len(self.atoms), 3), dtype=np.float64)
        return np.array([], dtype=np.float64).reshape(0, 3)
    
    def get_stress(self, include_ideal_gas=False):
        """Return stress tensor - MUST return 6-element array with dtype=float64"""
        return np.zeros(6, dtype=np.float64)
    
    def get_property(self, name, atoms=None):
        """Generic property getter for ASE compatibility"""
        if name == 'energy':
            return self.get_potential_energy()
        elif name == 'forces':
            return self.get_forces()
        elif name == 'stress':
            return self.get_stress()
        return None

# ============================================================================
# TEST STRUCTURE CREATION
# ============================================================================
st.header("1️⃣ Test Structure Creation")

structure_type = st.selectbox(
    "Select test structure",
    ["Bulk Sn (BCT)", "Bulk Li (BCC)", "Simple H₂ molecule"]
)

if structure_type == "Bulk Sn (BCT)":
    with st.spinner("Creating β-Sn structure..."):
        atoms = bulk('Sn', 'bct', a=5.83, c=3.18)
        st.success(f"✅ Created β-Sn: {len(atoms)} atoms, V = {atoms.get_volume():.2f} Å³")
        st.code(f"Cell: {atoms.get_cell().array}")
        st.code(f"Positions: {atoms.get_positions()}")
        
elif structure_type == "Bulk Li (BCC)":
    with st.spinner("Creating Li structure..."):
        atoms = bulk('Li', 'bcc', a=3.51)
        st.success(f"✅ Created Li: {len(atoms)} atoms, V = {atoms.get_volume():.2f} Å³")
        st.code(f"Cell: {atoms.get_cell().array}")
        
elif structure_type == "Simple H₂ molecule":
    with st.spinner("Creating H₂ molecule..."):
        atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.74)])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(False)
        st.success(f"✅ Created H₂: {len(atoms)} atoms")
        st.code(f"Positions: {atoms.get_positions()}")

# Store atoms in session state for later use
st.session_state['test_atoms'] = atoms

# ============================================================================
# GPAW CALCULATOR SETUP
# ============================================================================
st.header("2️⃣ GPAW Calculator Setup")

ecut = st.slider("Plane-wave cutoff (eV)", 100, 600, 350, 50)
kpts = st.selectbox("k-point grid", [(2,2,2), (3,3,3), (4,4,4), (6,6,6)], index=1)
xc = st.selectbox("Exchange-correlation functional", ["PBE", "LDA", "BLYP"], index=0)

st.markdown("### Calculator Parameters:")
st.json({
    "mode": f"PW(ecut={ecut} eV)",
    "xc": xc,
    "kpts": kpts,
    "convergence": {"energy": 1e-5, "density": 1e-4}
})

# ============================================================================
# CALCULATOR ATTACHMENT TEST
# ============================================================================
st.header("3️⃣ Calculator Attachment Test")

if st.button("🔧 Attach GPAW Calculator to Atoms", use_container_width=True):
    try:
        # Create GPAW calculator
        if GPAW_AVAILABLE:
            # Use real GPAW
            calc = GPAW(
                mode=PW(ecut),
                xc=xc,
                kpts=kpts,
                txt='gpaw_test.txt',
                convergence={'energy': 1e-5, 'density': 1e-4},
                maxiter=50,
                occupations={'name': 'fermi-dirac', 'width': 0.1}
            )
        else:
            # Use stub calculator
            st.warning("⚠️ Using GPAW stub calculator (real GPAW not available)")
            calc = GPAW_Stub(
                mode=PW(ecut),
                xc=xc,
                kpts=kpts,
                convergence={'energy': 1e-5, 'density': 1e-4},
                maxiter=50
            )
        
        # Attach calculator to atoms
        atoms = st.session_state['test_atoms']
        atoms.calc = calc
        
        st.success("✅ GPAW calculator successfully attached to Atoms object!")
        st.info(f"Calculator type: {type(atoms.calc).__name__}")
        st.info(f"Calculator module: {type(atoms.calc).__module__}")
        
        # 🔧 FIXED: Verify calculator properties with safe attribute checking
        st.markdown("### Calculator Properties:")
        calc_properties = {}
        
        # Check if calculator has atoms attribute
        calc_properties['has_atoms'] = hasattr(atoms.calc, 'atoms') and atoms.calc.atoms is not None
        
        # 🔧 FIXED: Safely check xc attribute
        if hasattr(atoms.calc, 'xc'):
            calc_properties['xc_functional'] = atoms.calc.xc
        else:
            calc_properties['xc_functional'] = 'N/A (attribute not found)'
        
        # Check kpts attribute
        if hasattr(atoms.calc, 'kpts'):
            calc_properties['kpts'] = str(atoms.calc.kpts)
        else:
            calc_properties['kpts'] = 'N/A'
        
        # Check ecut attribute
        if hasattr(atoms.calc, 'mode') and hasattr(atoms.calc.mode, 'ecut'):
            calc_properties['ecut'] = f"{atoms.calc.mode.ecut} eV"
        elif hasattr(atoms.calc, 'ecut'):
            calc_properties['ecut'] = f"{atoms.calc.ecut} eV"
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
        1. GPAW stub class missing required attributes
        2. Calculator not fully initialized before attachment
        3. Attribute accessed before being set
        """)
        st.exception(e)
    except Exception as e:
        st.error(f"❌ Failed to attach calculator: {e}")
        st.exception(e)
        st.session_state['calc_attached'] = False

# ============================================================================
# ENERGY CALCULATION TEST
# ============================================================================
st.header("4️⃣ Energy Calculation Test")

if st.session_state.get('calc_attached', False):
    if st.button("⚡ Calculate Single-Point Energy", use_container_width=True):
        try:
            atoms = st.session_state['test_atoms']
            
            with st.spinner("Running calculation..."):
                # Test 1: Check if calculator has required methods
                st.markdown("### Method Availability Check:")
                methods_to_check = ['get_potential_energy', 'get_forces', 'get_stress']
                for method in methods_to_check:
                    has_method = hasattr(atoms.calc, method)
                    st.success(f"✅ {method}: {has_method}") if has_method else st.error(f"❌ {method}: {has_method}")
                
                # Test 2: Calculate energy
                st.markdown("### Energy Calculation:")
                energy = atoms.get_potential_energy()
                st.metric("Total Energy", f"{energy:.6f} eV")
                st.metric("Energy per Atom", f"{energy/len(atoms):.6f} eV/atom")
                
                # Test 3: Calculate forces
                st.markdown("### Forces:")
                forces = atoms.get_forces()
                st.metric("Max Force", f"{np.abs(forces).max():.6f} eV/Å")
                st.metric("Avg Force", f"{np.abs(forces).mean():.6f} eV/Å")
                st.dataframe(
                    pd.DataFrame(forces, columns=['Fx', 'Fy', 'Fz']),
                    use_container_width=True
                )
                
                # Test 4: Calculate stress (if periodic)
                if atoms.pbc.any():
                    st.markdown("### Stress Tensor:")
                    stress = atoms.get_stress()
                    st.code(f"Stress (Voigt): {stress} eV/Å³")
                    stress_gpa = stress * 160.217  # Convert to GPa
                    st.code(f"Stress (GPa): {stress_gpa}")
                
                st.success("✅ All calculator methods working correctly!")
                
        except AttributeError as e:
            st.error(f"❌ AttributeError: {e}")
            st.markdown("**This indicates the calculator is not properly attached**")
            st.info("Try: `atoms.calc = calc` before calling `atoms.get_potential_energy()`")
            st.exception(e)
        except Exception as e:
            st.error(f"❌ Calculation failed: {e}")
            st.exception(e)
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
    st.metric("ASE Version", st.session_state.get('ase_version', 'Check below'))

# Check ASE version
try:
    import ase
    st.session_state['ase_version'] = ase.__version__
except:
    st.session_state['ase_version'] = "Unknown"

# Display detailed version info
with st.expander("📋 Detailed Version Information"):
    st.markdown("### Python Packages:")
    packages = {
        'streamlit': 'streamlit',
        'ase': 'ase',
        'gpaw': 'gpaw',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'plotly': 'plotly'
    }
    for pkg_name, import_name in packages.items():
        try:
            pkg = __import__(import_name)
            version = getattr(pkg, '__version__', 'Unknown')
            st.success(f"✅ {pkg_name}: {version}")
        except ImportError:
            st.error(f"❌ {pkg_name}: Not installed")

# ============================================================================
# TROUBLESHOOTING GUIDE
# ============================================================================
st.header("🔧 Troubleshooting Guide")

with st.expander("Common Issues & Solutions"):
    st.markdown("""
    ### Issue 1: `AttributeError: 'GPAW' object has no attribute 'xc'`
    **Cause**: GPAW stub class missing xc attribute initialization
    
    **Solution**:
    ```python
    class GPAW_Stub:
        def __init__(self, xc='PBE', ...):
            self.xc = xc  # ← Must store this attribute!
    ```
    
    ### Issue 2: `AttributeError: 'NoneType' object has no attribute 'get_forces'`
    **Cause**: Calculator not attached to atoms before optimization
    
    **Solution**:
    ```python
    atoms.calc = calc  # ← Must do this BEFORE optimization
    opt = BFGS(atoms)
    opt.run(fmax=0.05)
    ```
    
    ### Issue 3: `ImportError: No module named 'gpaw'`
    **Cause**: GPAW not installed
    
    **Solution**:
    ```bash
    pip install gpaw
    # Or with conda:
    conda install -c conda-forge gpaw
    ```
    
    ### Issue 4: Calculator methods return empty arrays
    **Cause**: DummyCalculator instead of real GPAW
    
    **Solution**: Check `GPAW_AVAILABLE` flag and ensure GPAW import succeeds
    
    ### Issue 5: Optimization fails with `get_forces()` error
    **Cause**: Calculator doesn't implement required ASE interface
    
    **Solution**: Ensure calculator inherits from `ase.calculators.calculator.Calculator`
    or implements all required methods: get_potential_energy(), get_forces(), get_stress()
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
<strong>GPAW + ASE Integration Test</strong><br>
Use this app to verify your DFT backend is working before running full analysis<br>
<em>Version 2.0.1 - Fixed GPAW Stub Class Attributes</em>
</div>
""", unsafe_allow_html=True)

if st.session_state.get('last_error') and st.session_state.get('enable_detailed_logging', False):
    with st.expander("🐛 Last Error Details (Debug)", expanded=False):
        st.code(st.session_state['last_error'], language="text")
        if st.button("Clear Error", key="clear_err"):
            st.session_state['last_error'] = None
            st.rerun()
