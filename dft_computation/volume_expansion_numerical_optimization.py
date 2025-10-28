import streamlit as st
from ase import Atoms
from ase.optimize import BFGS, LBFGS
from ase.spacegroup import crystal
from ase.constraints import ExpCellFilter
from gpaw import GPAW, PW
import os

st.title("ASE + GPAW: Fast vc-relax for Li-Sn System")

st.markdown("""
This app performs **fast** variable-cell relaxation (vc-relax) for Lithium-Tin systems using ASE + GPAW.
**Optimized for testing and prototyping** with adjustable accuracy settings.

**Speed vs Accuracy Trade-offs:**
- Lower cutoff energy → Faster but less accurate
- Fewer k-points → Much faster but less accurate  
- Looser convergence → Faster convergence but less precise
""")

# Speed optimization controls
st.sidebar.header("⚡ Speed Optimization")
calculation_mode = st.sidebar.selectbox(
    "Calculation Mode",
    options=["Fast Testing", "Balanced", "High Accuracy"],
    index=0,
    help="Choose between speed and accuracy"
)

# Set parameters based on calculation mode
if calculation_mode == "Fast Testing":
    default_energy_conv = 1e-4
    default_force_conv = 5e-3
    ecut_factor = 0.7
    kpts_factor = 0.5
    max_steps = 50
elif calculation_mode == "Balanced":
    default_energy_conv = 1e-5
    default_force_conv = 1e-3
    ecut_factor = 0.85
    kpts_factor = 0.75
    max_steps = 100
else:  # High Accuracy
    default_energy_conv = 1e-6
    default_force_conv = 1e-4
    ecut_factor = 1.0
    kpts_factor = 1.0
    max_steps = 200

st.sidebar.write(f"**Current Mode:** {calculation_mode}")
st.sidebar.write(f"**Max Steps:** {max_steps}")

# Select structure
structure = st.selectbox("Select Structure", options=['Li (BCC)', 'Sn (diamond cubic)', 'Sn (BCT)', 'Li2Sn5'])

# Default parameters based on structure (high accuracy baseline)
if structure == 'Li (BCC)':
    default_ka_kb = int(10 * kpts_factor)
    default_kc = int(10 * kpts_factor)
    default_ecut = int(400 * ecut_factor)
    default_a = 3.49
elif structure == 'Sn (diamond cubic)':
    default_ka_kb = int(8 * kpts_factor)
    default_kc = int(8 * kpts_factor)
    default_ecut = int(500 * ecut_factor)
    default_a = 6.49
elif structure == 'Sn (BCT)':
    default_ka_kb = int(8 * kpts_factor)
    default_kc = int(12 * kpts_factor)
    default_ecut = int(500 * ecut_factor)
    default_a = 5.83
    default_c = 3.18
elif structure == 'Li2Sn5':
    default_ka_kb = int(6 * kpts_factor)
    default_kc = int(16 * kpts_factor)
    default_ecut = int(500 * ecut_factor)
    default_a = 10.274
    default_c = 3.125

# Initial parameters with speed-optimized defaults
if structure == 'Li (BCC)':
    a = st.number_input("Initial a (Å)", min_value=2.0, max_value=10.0, value=default_a)
    atoms = crystal('Li', basis=[(0,0,0)], spacegroup=229, cellpar=[a, a, a, 90, 90, 90])
    num_sn = 0
    is_cubic = True
elif structure == 'Sn (diamond cubic)':
    a = st.number_input("Initial a (Å)", min_value=4.0, max_value=10.0, value=default_a)
    atoms = crystal('Sn', basis=[(0,0,0), (0.25,0.25,0.25)], spacegroup=227, cellpar=[a, a, a, 90, 90, 90])
    num_sn = 8
    is_cubic = True
elif structure == 'Sn (BCT)':
    a = st.number_input("Initial a (Å)", min_value=2.0, max_value=10.0, value=default_a)
    c = st.number_input("Initial c (Å)", min_value=2.0, max_value=10.0, value=default_c)
    atoms = crystal('Sn', basis=[(0,0,0)], spacegroup=141, cellpar=[a, a, c, 90, 90, 90])
    num_sn = len(atoms)  # 4
    is_cubic = False
elif structure == 'Li2Sn5':
    a = st.number_input("Initial a (Å)", min_value=5.0, max_value=15.0, value=default_a)
    c = st.number_input("Initial c (Å)", min_value=2.0, max_value=5.0, value=default_c)
    atoms = crystal(
        symbols=['Sn', 'Li', 'Sn'],
        basis=[(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)],
        spacegroup=127,
        cellpar=[a, a, c, 90, 90, 90]
    )
    num_sn = 10
    is_cubic = False

# Convergence thresholds with speed-optimized defaults
col1, col2 = st.columns(2)

with col1:
    etot_conv_thr = st.number_input(
        "SCF energy convergence threshold (eV/atom)",
        min_value=1e-8,
        max_value=1e-2,
        value=default_energy_conv,
        format="%.1e",
        help="Larger values = faster but less accurate"
    )
    
    ecut = st.number_input(
        "Plane-wave cutoff energy (eV)",
        min_value=200,  # Lower minimum for testing
        max_value=1000,
        value=default_ecut,
        help="Lower values = faster but less accurate"
    )

with col2:
    forc_conv_thr = st.number_input(
        "Force convergence threshold (eV/Å)",
        min_value=1e-8,
        max_value=1e-1,  # Allow larger values for speed
        value=default_force_conv,
        format="%.1e",
        help="Larger values = faster convergence"
    )
    
    # K-points inputs
    ka_kb = st.number_input(
        "K-points along a/b (ka x kb)",
        min_value=2,  # Lower minimum
        max_value=20,
        value=default_ka_kb,
        help="Fewer k-points = much faster"
    )
    kc = st.number_input(
        "K-points along c (kc)",
        min_value=2,
        max_value=20,
        value=default_kc
    )

# Additional speed optimizations
st.sidebar.subheader("Advanced Speed Options")
use_fast_optimizer = st.sidebar.checkbox("Use Fast Optimizer (LBFGS)", value=True)
reduce_mixer = st.sidebar.checkbox("Reduce Mixer Settings", value=True)
simple_occupations = st.sidebar.checkbox("Simple Occupations", value=True)

# Display current speed settings
st.sidebar.markdown("---")
st.sidebar.subheader("Current Speed Settings")
st.sidebar.write(f"ECut: {ecut} eV")
st.sidebar.write(f"K-points: ({ka_kb}, {ka_kb}, {kc})")
st.sidebar.write(f"Force convergence: {forc_conv_thr:.1e} eV/Å")
st.sidebar.write(f"Energy convergence: {etot_conv_thr:.1e} eV/atom")

def create_fast_calculator(atoms, structure_name, ecut, ka_kb, kc):
    """Create optimized calculator for speed"""
    convergence_settings = {'energy': etot_conv_thr}
    
    # Additional speed optimizations
    if reduce_mixer:
        from gpaw import Mixer
        mixer = Mixer(0.1, 5, 10)  # Faster mixing
    else:
        mixer = None
        
    if simple_occupations:
        occupations = None  # Default occupations (faster for metals)
    else:
        from gpaw import FermiDirac
        occupations = FermiDirac(0.1)
    
    calculator = GPAW(
        mode=PW(ecut),
        xc='PBE',
        kpts=(ka_kb, ka_kb, kc),
        convergence=convergence_settings,
        txt=f'{structure_name.replace(" ", "_")}_gpaw.log',
        maxiter=100,  # Reduced electronic steps
        mixer=mixer,
        occupations=occupations,
    )
    
    return calculator

run_calc = st.button("🚀 Run Fast vc-relax Calculation")

if run_calc:
    try:
        st.write(f"Setting up **fast** vc-relax for {structure}...")
        st.write(f"**Mode:** {calculation_mode}")
        
        # Show optimization parameters
        with st.expander("Current Optimization Parameters"):
            st.write(f"- Plane-wave cutoff: {ecut} eV")
            st.write(f"- K-points grid: ({ka_kb}, {ka_kb}, {kc})")
            st.write(f"- Force convergence: {forc_conv_thr:.1e} eV/Å")
            st.write(f"- Energy convergence: {etot_conv_thr:.1e} eV/atom")
            st.write(f"- Maximum steps: {max_steps}")

        # Set GPAW calculator with fast settings
        calc = create_fast_calculator(atoms, structure, ecut, ka_kb, kc)
        atoms.calc = calc

        st.write("Starting variable-cell relaxation...")

        # Use ExpCellFilter for variable-cell relaxation
        ecf = ExpCellFilter(atoms)
        
        # Choose optimizer
        if use_fast_optimizer:
            opt = LBFGS(ecf, logfile=f'{structure}_relax.log')  # LBFGS is generally faster
        else:
            opt = BFGS(ecf, logfile=f'{structure}_relax.log')

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress():
            if opt.nsteps > 0:
                progress = min(opt.nsteps / max_steps, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Optimization step {opt.nsteps}/{max_steps}")

        opt.attach(update_progress, 1)
        
        # Run with step limit
        opt.run(fmax=forc_conv_thr, steps=max_steps)

        # Get results
        final_cell = atoms.get_cell()
        lattice_constants_angstrom = final_cell.lengths()
        total_energy = atoms.get_potential_energy()
        volume = atoms.get_volume()
        final_forces = atoms.get_forces()
        max_force = np.max(np.abs(final_forces))

        st.success(f"Relaxation complete for {structure}!")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Structural Results")
            st.write(f"**Optimized lattice constants (Å):** {lattice_constants_angstrom}")
            st.write(f"**Volume (Å³):** {volume:.6f}")
            st.write(f"**Final max force:** {max_force:.6f} eV/Å")
            
        with col2:
            st.subheader("Energetic Results")
            st.write(f"**Total energy (eV):** {total_energy:.6f}")
            st.write(f"**Energy/atom (eV):** {total_energy/len(atoms):.6f}")
            st.write(f"**Convergence:** {'Yes' if max_force <= forc_conv_thr else 'No'}")

        if max_force > forc_conv_thr:
            st.warning("Calculation stopped before full convergence. Consider increasing max steps or loosening convergence criteria.")

    except Exception as e:
        st.error(f"Calculation failed: {e}")

# Fast volume expansion calculation
st.header("⚡ Fast Volume Expansion Analysis")

def run_fast_expansion():
    """Run optimized volume expansion calculation"""
    try:
        # Use current calculation mode settings for expansion
        if calculation_mode == "Fast Testing":
            ecut_exp = 350
            kpts_sn = (4, 4, 6)  # Reduced k-points
            kpts_li2sn5 = (3, 3, 8)
            forc_conv_exp = 5e-3
            max_steps_exp = 30
        elif calculation_mode == "Balanced":
            ecut_exp = 450
            kpts_sn = (6, 6, 8)
            kpts_li2sn5 = (4, 4, 12)
            forc_conv_exp = 1e-3
            max_steps_exp = 60
        else:
            ecut_exp = 500
            kpts_sn = (8, 8, 12)
            kpts_li2sn5 = (6, 6, 16)
            forc_conv_exp = 1e-4
            max_steps_exp = 100

        st.write(f"### Phase 1: Fast BCT Sn Relaxation")
        st.write(f"Using k-points: {kpts_sn}, Ecut: {ecut_exp} eV")
        
        # BCT Sn
        a_sn = 5.83
        c_sn = 3.18
        atoms_sn = crystal('Sn', basis=[(0,0,0)], spacegroup=141, cellpar=[a_sn, a_sn, c_sn, 90, 90, 90])
        
        calc_sn = GPAW(
            mode=PW(ecut_exp),
            xc='PBE',
            kpts=kpts_sn,
            convergence={'energy': etot_conv_thr},
            txt='Sn_BCT_fast.log',
            maxiter=80
        )
        atoms_sn.calc = calc_sn
        ecf_sn = ExpCellFilter(atoms_sn)
        opt_sn = LBFGS(ecf_sn, logfile='Sn_BCT_fast_relax.log')
        opt_sn.run(fmax=forc_conv_exp, steps=max_steps_exp)
        
        num_sn_atoms = len(atoms_sn)
        v_sn = atoms_sn.get_volume() / num_sn_atoms

        st.write(f"### Phase 2: Fast Li₂Sn₅ Relaxation") 
        st.write(f"Using k-points: {kpts_li2sn5}, Ecut: {ecut_exp} eV")
        
        # Li₂Sn₅
        a_li2sn5 = 10.274
        c_li2sn5 = 3.125
        atoms_li2sn5 = crystal(
            symbols=['Sn', 'Li', 'Sn'],
            basis=[(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)],
            spacegroup=127,
            cellpar=[a_li2sn5, a_li2sn5, c_li2sn5, 90, 90, 90]
        )
        
        calc_li2sn5 = GPAW(
            mode=PW(ecut_exp),
            xc='PBE',
            kpts=kpts_li2sn5,
            convergence={'energy': etot_conv_thr},
            txt='Li2Sn5_fast.log',
            maxiter=80
        )
        atoms_li2sn5.calc = calc_li2sn5
        ecf_li2sn5 = ExpCellFilter(atoms_li2sn5)
        opt_li2sn5 = LBFGS(ecf_li2sn5, logfile='Li2Sn5_fast_relax.log')
        opt_li2sn5.run(fmax=forc_conv_exp, steps=max_steps_exp)
        
        num_sn_li2sn5 = sum(1 for atom in atoms_li2sn5 if atom.symbol == 'Sn')
        v_li2sn5 = atoms_li2sn5.get_volume() / num_sn_li2sn5

        expansion = (v_li2sn5 - v_sn) / v_sn * 100

        st.success("### Fast Volume Expansion Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("BCT Sn Volume/Sn", f"{v_sn:.4f} Å³")
            st.metric("Li₂Sn₅ Volume/Sn", f"{v_li2sn5:.4f} Å³")
        with col2:
            st.metric("Volume Expansion", f"{expansion:.2f}%", delta=f"{expansion:.2f}%")
        
        st.info(f"**Calculation Mode:** {calculation_mode}")
        st.info("For production runs, use 'High Accuracy' mode")

    except Exception as e:
        st.error(f"Fast expansion calculation failed: {e}")

run_fast_expansion = st.button("🚀 Compute Fast Volume Expansion")

if run_fast_expansion:
    st.info(f"Starting fast volume expansion in {calculation_mode} mode...")
    run_fast_expansion()

# Tips for further speedup
with st.expander("💡 Additional Speed Optimization Tips"):
    st.markdown("""
    **For Maximum Speed (Testing Only):**
    - Use **ECut = 300 eV** and **k-points = (2,2,2)**
    - Set **force convergence = 0.01 eV/Å** 
    - Set **energy convergence = 1e-3 eV/atom**
    - Use **LBFGS optimizer** with **max steps = 20**
    
    **Expected Speedup:** 5-10x faster than high-accuracy settings
    
    **Warning:** Results will be qualitative only - use for initial testing and prototyping
    """)

# Performance estimates
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Performance Estimates")
if calculation_mode == "Fast Testing":
    st.sidebar.write("**Estimate:** 2-10 minutes")
    st.sidebar.write("**Use case:** Quick testing")
elif calculation_mode == "Balanced":
    st.sidebar.write("**Estimate:** 10-60 minutes") 
    st.sidebar.write("**Use case:** Development")
else:
    st.sidebar.write("**Estimate:** 1-6 hours")
    st.sidebar.write("**Use case:** Production")
