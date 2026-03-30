#!/usr/bin/env python3
"""
GPAW + ASE Integration Test
============================
Simple Streamlit app to verify GPAW calculator works with ASE
Run with: streamlit run gpaw_test.py
"""

import streamlit as st
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS

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
        calc = GPAW(
            mode=PW(ecut),
            xc=xc,
            kpts=kpts,
            txt='gpaw_test.txt',
            convergence={'energy': 1e-5, 'density': 1e-4},
            maxiter=50,
            occupations={'name': 'fermi-dirac', 'width': 0.1}
        )
        
        # Attach calculator to atoms
        atoms.calc = calc
        
        st.success("✅ GPAW calculator successfully attached to Atoms object!")
        st.info(f"Calculator type: {type(atoms.calc).__name__}")
        st.info(f"Calculator module: {type(atoms.calc).__module__}")
        
        # Verify calculator properties
        st.markdown("### Calculator Properties:")
        st.json({
            "has_atoms": atoms.calc.atoms is not None,
            "xc_functional": atoms.calc.xc,
            "kpts": atoms.calc.kpts,
            "ecut": atoms.calc.mode.ecut if hasattr(atoms.calc.mode, 'ecut') else "N/A"
        })
        
        # Save calculator state to session
        st.session_state['calc_attached'] = True
        st.session_state['atoms'] = atoms
        st.session_state['calc'] = calc
        
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
            atoms = st.session_state['atoms']
            
            with st.spinner("Running GPAW calculation..."):
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
                
                st.success("✅ All GPAW calculator methods working correctly!")
                
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
# OPTIMIZATION TEST
# ============================================================================
st.header("5️⃣ Geometry Optimization Test (BFGS)")

if st.session_state.get('calc_attached', False):
    if st.button("🔬 Run Geometry Optimization", use_container_width=True):
        try:
            atoms = st.session_state['atoms'].copy()
            
            # Re-attach calculator to copied atoms
            calc = GPAW(
                mode=PW(ecut),
                xc=xc,
                kpts=kpts,
                txt='gpaw_optimize.txt',
                convergence={'energy': 1e-5, 'density': 1e-4},
                maxiter=100
            )
            atoms.calc = calc
            
            with st.spinner("Optimizing geometry..."):
                opt = BFGS(atoms, logfile=None)
                converged = opt.run(fmax=0.05, steps=50)
                
                st.success(f"✅ Optimization {'converged' if converged else 'did not converge'}")
                st.metric("Final Energy", f"{atoms.get_potential_energy():.6f} eV")
                st.metric("Max Force", f"{np.abs(atoms.get_forces()).max():.6f} eV/Å")
                
                st.markdown("### Final Structure:")
                st.code(f"Cell: {atoms.get_cell().array}")
                st.code(f"Positions: {atoms.get_positions()}")
                
        except AttributeError as e:
            st.error(f"❌ AttributeError during optimization: {e}")
            st.markdown("**Common cause: Calculator not attached before optimization**")
            st.info("Ensure `atoms.calc = calc` is called before `BFGS(atoms).run()`")
            st.exception(e)
        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")
            st.exception(e)

# ============================================================================
# DIAGNOSTIC INFORMATION
# ============================================================================
st.header("📊 System Information")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("GPAW Available", "✅ Yes" if GPAW_AVAILABLE else "❌ No")
    st.metric("GPAW Version", GPAW_VERSION or "N/A")
with col2:
    import sys
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
    ### Issue 1: `AttributeError: 'NoneType' object has no attribute 'get_forces'`
    **Cause**: Calculator not attached to atoms before optimization
    
    **Solution**:
    ```python
    atoms.calc = calc  # ← Must do this BEFORE optimization
    opt = BFGS(atoms)
    opt.run(fmax=0.05)
    ```
    
    ### Issue 2: `ImportError: No module named 'gpaw'`
    **Cause**: GPAW not installed
    
    **Solution**:
    ```bash
    pip install gpaw
    # Or with conda:
    conda install -c conda-forge gpaw
    ```
    
    ### Issue 3: Calculator methods return empty arrays
    **Cause**: DummyCalculator instead of real GPAW
    
    **Solution**: Check `GPAW_AVAILABLE` flag and ensure GPAW import succeeds
    
    ### Issue 4: Optimization fails with `get_forces()` error
    **Cause**: Calculator doesn't implement required ASE interface
    
    **Solution**: Ensure calculator inherits from `ase.calculators.calculator.Calculator`
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
<strong>GPAW + ASE Integration Test</strong><br>
Use this app to verify your DFT backend is working before running full analysis
</div>
""", unsafe_allow_html=True)
