import streamlit as st
import numpy as np
from ase import Atoms
from ase.optimize import BFGS, LBFGS
from ase.spacegroup import crystal
from ase.constraints import UnitCellFilter
from ase.io import write
import ase.units as units
from gpaw import GPAW, PW, FermiDirac, Mixer, MixerSum
from gpaw.eigensolvers import RMM-DIIS
import os
import tempfile
from contextlib import contextmanager
import traceback

# Scientific constants
BOHR_TO_ANGSTROM = 0.529177210903
EV_TO_HARTREE = 1.0 / 27.211384523

st.title("ASE + GPAW: Advanced vc-relax for Li-Sn System with Volume Expansion Analysis")

st.markdown("""
This app performs robust variable-cell relaxation (vc-relax) for Lithium-Tin systems using ASE + GPAW with enhanced numerical stability.
Features include:
- Advanced convergence controls
- Multiple optimization algorithms
- Comprehensive volume expansion analysis
- Robust error handling and validation
""")

# Configuration section
st.sidebar.header("Calculation Settings")

# Advanced calculator parameters
st.sidebar.subheader("DFT Parameters")
xc_functional = st.sidebar.selectbox(
    "XC Functional",
    options=['PBE', 'PBEsol', 'LDA'],
    index=0,
    help="Exchange-correlation functional"
)

convergence_mode = st.sidebar.selectbox(
    "Convergence Mode",
    options=['Standard', 'High Precision', 'Fast'],
    index=0
)

# Set convergence parameters based on mode
if convergence_mode == 'High Precision':
    energy_conv = 1e-7
    force_conv = 1e-5
    stress_conv = 1e-6
    electronic_conv = 1e-9
elif convergence_mode == 'Fast':
    energy_conv = 1e-5
    force_conv = 1e-3
    stress_conv = 1e-4
    electronic_conv = 1e-6
else:  # Standard
    energy_conv = 1e-6
    force_conv = 1e-4
    stress_conv = 1e-5
    electronic_conv = 1e-8

# Main calculation parameters
etot_conv_thr = st.number_input(
    "Total energy convergence threshold (eV)",
    min_value=1e-9,
    max_value=1e-3,
    value=energy_conv,
    format="%.1e",
    help="Convergence threshold for total energy"
)

forc_conv_thr = st.number_input(
    "Force convergence threshold (eV/Å)",
    min_value=1e-9,
    max_value=1e-2,
    value=force_conv,
    format="%.1e",
    help="Convergence threshold for atomic forces"
)

stress_conv_thr = st.number_input(
    "Stress convergence threshold (GPa)",
    min_value=1e-6,
    max_value=1.0,
    value=stress_conv,
    format="%.1e",
    help="Convergence threshold for stress tensor"
)

ecut = st.number_input(
    "Plane-wave cutoff energy (eV)",
    min_value=100,
    max_value=1500,
    value=600 if 'Sn' in structure else 450,
    step=50,
    help="Plane-wave energy cutoff. Higher for heavier elements"
)

kpts_density = st.number_input(
    "K-points density (per Å⁻¹)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.1,
    help="K-point density for automatic mesh generation"
)

# Structure selection with enhanced parameters
structure = st.selectbox(
    "Select Structure", 
    options=['Li (cubic)', 'Sn (cubic)', 'Sn (BCT)', 'Li2Sn5'],
    help="Select crystal structure for relaxation"
)

# Enhanced structure definitions with validation
def create_li_cubic():
    alat = st.number_input(
        "Initial lattice constant (alat, Bohr)",
        min_value=3.0,
        max_value=8.0,
        value=6.0,
        step=0.1,
        help="Lithium lattice constant in Bohr units"
    )
    alat_angstrom = alat * BOHR_TO_ANGSTROM
    atoms = Atoms(
        'Li',
        positions=[(0, 0, 0)],
        cell=np.eye(3) * alat_angstrom,
        pbc=True
    )
    return atoms, 1, "Li cubic"

def create_sn_cubic():
    alat = st.number_input(
        "Initial lattice constant (alat, Bohr)",
        min_value=6.0,
        max_value=10.0,
        value=7.5,
        step=0.1
    )
    alat_angstrom = alat * BOHR_TO_ANGSTROM
    atoms = Atoms(
        'Sn',
        positions=[(0, 0, 0)],
        cell=np.eye(3) * alat_angstrom,
        pbc=True
    )
    return atoms, 1, "Sn cubic"

def create_sn_bct():
    a = st.number_input("Initial a (Å)", min_value=5.0, max_value=7.0, value=5.831, step=0.001)
    c = st.number_input("Initial c (Å)", min_value=2.5, max_value=4.0, value=3.181, step=0.001)
    c_over_a = c / a
    st.write(f"c/a ratio: {c_over_a:.4f}")
    
    atoms = crystal(
        'Sn',
        basis=[(0, 0, 0)],
        spacegroup=141,  # I4₁/amd
        cellpar=[a, a, c, 90, 90, 90]
    )
    return atoms, len(atoms), "Sn BCT"

def create_li2sn5():
    st.markdown("**Li₂Sn₅ Crystal Structure (P4/mbm)**")
    a = st.number_input("Initial a (Å)", min_value=9.0, max_value=12.0, value=10.274, step=0.001)
    c = st.number_input("Initial c (Å)", min_value=2.5, max_value=4.0, value=3.125, step=0.001)
    
    # Accurate atomic positions for Li₂Sn₅ (P4/mbm, No. 127)
    # Reference: Hansen & Chang, 1969
    positions = []
    symbols = []
    
    # Sn atoms (10 total)
    # 2 Sn at 2a sites (0,0,0) and (1/2,1/2,0)
    positions.extend([(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)])
    symbols.extend(['Sn', 'Sn'])
    
    # 8 Sn at 8i sites (x,y,0) with x=0.295, y=0.432
    x_sn, y_sn = 0.295, 0.432
    sn_positions = [
        (x_sn, y_sn, 0.0), (1-x_sn, 1-y_sn, 0.0),
        (-x_sn, -y_sn, 0.0), (1+x_sn, 1+y_sn, 0.0),
        (-y_sn, x_sn, 0.0), (y_sn, 1-x_sn, 0.0),
        (1-y_sn, 1+x_sn, 0.0), (1+y_sn, x_sn, 0.0)
    ]
    positions.extend(sn_positions)
    symbols.extend(['Sn'] * 8)
    
    # Li atoms (4 total)
    # 4 Li at 4g sites (x,1/2+x,0) with x=0.16
    x_li = 0.16
    li_positions = [
        (x_li, 0.5+x_li, 0.0), (1-x_li, 0.5-x_li, 0.0),
        (0.5-x_li, x_li, 0.0), (0.5+x_li, 1-x_li, 0.0)
    ]
    positions.extend(li_positions)
    symbols.extend(['Li'] * 4)
    
    # Create cell and scale positions
    cell = np.array([[a, 0, 0], [0, a, 0], [0, 0, c]])
    scaled_positions = np.array(positions)
    
    atoms = Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=True
    )
    
    return atoms, 10, "Li₂Sn₅"  # 10 Sn atoms for volume calculation

# Structure creation
if structure == 'Li (cubic)':
    atoms, num_sn, structure_name = create_li_cubic()
elif structure == 'Sn (cubic)':
    atoms, num_sn, structure_name = create_sn_cubic()
elif structure == 'Sn (BCT)':
    atoms, num_sn, structure_name = create_sn_bct()
elif structure == 'Li2Sn5':
    atoms, num_sn, structure_name = create_li2sn5()

# Display structure info
st.subheader("Structure Information")
st.write(f"Number of atoms: {len(atoms)}")
st.write(f"Initial volume: {atoms.get_volume():.6f} Å³")
st.write(f"Initial density: {len(atoms)/atoms.get_volume():.6f} atoms/Å³")

# Optimization settings
st.sidebar.subheader("Optimization Parameters")
optimizer_choice = st.sidebar.selectbox(
    "Optimization Algorithm",
    options=['BFGS', 'LBFGS', 'FIRE'],
    index=1,
    help="LBFGS is generally more efficient for variable-cell relaxation"
)

max_steps = st.sidebar.number_input(
    "Maximum optimization steps",
    min_value=10,
    max_value=500,
    value=100,
    help="Maximum number of relaxation steps"
)

trajectory_file = st.sidebar.checkbox("Save trajectory", value=True)

@contextmanager
def calculation_context(structure_name):
    """Context manager for robust calculation execution"""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        try:
            os.chdir(tmpdir)
            st.info(f"Running calculation in temporary directory: {tmpdir}")
            yield tmpdir
        except Exception as e:
            st.error(f"Calculation failed: {str(e)}")
            st.code(traceback.format_exc())
        finally:
            os.chdir(original_dir)

def create_advanced_calculator(atoms, structure_name, ecut, kpts_density, xc_functional):
    """Create a robust GPAW calculator with advanced settings"""
    
    # Automatic k-points generation based on cell size
    cell_volume = atoms.get_volume()
    kpts = max(4, int(kpts_density * (cell_volume ** (1/3))))
    
    calculator = GPAW(
        mode=PW(ecut),
        xc=xc_functional,
        kpts={'size': (kpts, kpts, kpts), 'gamma': True},
        convergence={
            'energy': electronic_conv,
            'density': 1e-6,
            'eigenstates': 1e-9,
            'bands': 'CBM+3.0'
        },
        occupations=FermiDirac(0.05),  # Smearing for metals
        mixer=MixerSum(0.05, 5, 50),   # Improved charge mixing
        eigensolver='rmm-diis',        # Robust eigensolver
        symmetry={'point_group': False}, # Disable symmetry for better convergence
        txt=f'{structure_name.replace(" ", "_")}_gpaw.log',
        maxiter=300,                   # Increased electronic steps
        dtype=complex,                 # Complex for better stability
        setups={'Sn': '4'},            # Specific setup for Sn
    )
    
    return calculator

def run_vc_relax(atoms, structure_name, forc_conv_thr, stress_conv_thr, max_steps):
    """Run variable-cell relaxation with enhanced robustness"""
    
    # Create calculator
    calc = create_advanced_calculator(atoms, structure_name, ecut, kpts_density, xc_functional)
    atoms.calc = calc
    
    # Use UnitCellFilter for variable-cell relaxation
    ucf = UnitCellFilter(atoms, mask=[1, 1, 1, 1, 1, 1])  # Allow all cell components to relax
    
    # Select optimizer
    if optimizer_choice == 'BFGS':
        opt = BFGS(ucf, logfile=f'{structure_name.replace(" ", "_")}_relax.log')
    elif optimizer_choice == 'LBFGS':
        opt = LBFGS(ucf, logfile=f'{structure_name.replace(" ", "_")}_relax.log')
    else:  # FIRE
        from ase.optimize import FIRE
        opt = FIRE(ucf, logfile=f'{structure_name.replace(" ", "_")}_relax.log')
    
    if trajectory_file:
        opt.attach(lambda: write(f'{structure_name.replace(" ", "_")}_traj.traj', atoms), 1)
    
    # Run optimization with progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress():
        step = opt.nsteps
        progress = min(step / max_steps, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Optimization step {step}/{max_steps}")
    
    opt.attach(update_progress, 1)
    
    try:
        opt.run(fmax=forc_conv_thr, steps=max_steps, smax=stress_conv_thr*units.GPa)
        progress_bar.progress(1.0)
        status_text.text("Optimization completed successfully!")
        return atoms, True
    except Exception as e:
        st.warning(f"Optimization stopped: {str(e)}")
        return atoms, False

# Main calculation execution
run_calc = st.button("Run Advanced vc-relax Calculation")

if run_calc:
    with calculation_context(structure_name):
        try:
            st.write(f"Starting advanced vc-relax for {structure_name}...")
            
            initial_volume = atoms.get_volume()
            initial_cell = atoms.cell.copy()
            
            # Run relaxation
            relaxed_atoms, success = run_vc_relax(
                atoms, structure_name, forc_conv_thr, stress_conv_thr, max_steps
            )
            
            if success:
                # Calculate results
                final_cell = relaxed_atoms.get_cell()
                lattice_constants = final_cell.lengths()
                angles = final_cell.angles()
                total_energy = relaxed_atoms.get_potential_energy()
                final_volume = relaxed_atoms.get_volume()
                volume_change = ((final_volume - initial_volume) / initial_volume) * 100
                
                # Stress analysis
                stress_tensor = relaxed_atoms.get_stress() / units.GPa
                pressure = -np.trace(stress_tensor[:3]) / 3
                
                st.success("Relaxation completed successfully!")
                
                # Display comprehensive results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Structural Results")
                    st.write(f"**Lattice constants (Å):** {lattice_constants}")
                    st.write(f"**Cell angles (°):** {angles}")
                    st.write(f"**Final volume:** {final_volume:.6f} Å³")
                    st.write(f"**Volume change:** {volume_change:+.4f}%")
                    st.write(f"**Density:** {len(relaxed_atoms)/final_volume:.6f} atoms/Å³")
                
                with col2:
                    st.subheader("Energetic Results")
                    st.write(f"**Total energy:** {total_energy:.8f} eV")
                    st.write(f"**Energy/atom:** {total_energy/len(relaxed_atoms):.8f} eV/atom")
                    st.write(f"**Pressure:** {pressure:.6f} GPa")
                    st.write(f"**Max stress component:** {np.max(np.abs(stress_tensor)):.6f} GPa")
                
                # Cell change analysis
                st.subheader("Cell Evolution")
                cell_change = np.linalg.norm(final_cell - initial_cell) / np.linalg.norm(initial_cell) * 100
                st.write(f"**Total cell change:** {cell_change:.4f}%")
                
            else:
                st.warning("Relaxation did not fully converge. Results may be approximate.")
                
        except Exception as e:
            st.error(f"Calculation failed: {str(e)}")
            st.code(traceback.format_exc())

# Enhanced volume expansion analysis
st.header("Volume Expansion Analysis: BCT Sn → Li₂Sn₅")

def calculate_volume_expansion():
    """Comprehensive volume expansion calculation with error handling"""
    
    with calculation_context("Volume_Expansion"):
        try:
            st.write("### Phase 1: BCT Sn Relaxation")
            
            # BCT Sn calculation
            atoms_sn_bct, num_sn_sn, _ = create_sn_bct()
            atoms_sn_bct.calc = create_advanced_calculator(
                atoms_sn_bct, "Sn_BCT", 600, 2.5, xc_functional
            )
            
            ucf_sn = UnitCellFilter(atoms_sn_bct)
            opt_sn = LBFGS(ucf_sn, logfile='Sn_BCT_relax.log')
            opt_sn.run(fmax=forc_conv_thr, smax=stress_conv_thr*units.GPa, steps=100)
            
            v_sn_per_atom = atoms_sn_bct.get_volume() / len(atoms_sn_bct)
            v_sn_per_sn = atoms_sn_bct.get_volume() / num_sn_sn
            
            st.write(f"BCT Sn volume per atom: {v_sn_per_atom:.6f} Å³")
            st.write(f"BCT Sn volume per Sn atom: {v_sn_per_sn:.6f} Å³")
            
            st.write("### Phase 2: Li₂Sn₅ Relaxation")
            
            # Li₂Sn₅ calculation
            atoms_li2sn5, num_sn_li2sn5, _ = create_li2sn5()
            atoms_li2sn5.calc = create_advanced_calculator(
                atoms_li2sn5, "Li2Sn5", 600, 2.0, xc_functional
            )
            
            ucf_li2sn5 = UnitCellFilter(atoms_li2sn5)
            opt_li2sn5 = LBFGS(ucf_li2sn5, logfile='Li2Sn5_relax.log')
            opt_li2sn5.run(fmax=forc_conv_thr, smax=stress_conv_thr*units.GPa, steps=150)
            
            v_li2sn5_total = atoms_li2sn5.get_volume()
            v_li2sn5_per_sn = v_li2sn5_total / num_sn_li2sn5
            v_li2sn5_per_atom = v_li2sn5_total / len(atoms_li2sn5)
            
            st.write(f"Li₂Sn₅ total volume: {v_li2sn5_total:.6f} Å³")
            st.write(f"Li₂Sn₅ volume per Sn atom: {v_li2sn5_per_sn:.6f} Å³")
            st.write(f"Li₂Sn₅ volume per atom: {v_li2sn5_per_atom:.6f} Å³")
            
            # Volume expansion calculations
            expansion_per_sn = ((v_li2sn5_per_sn - v_sn_per_sn) / v_sn_per_sn) * 100
            expansion_per_atom = ((v_li2sn5_per_atom - v_sn_per_atom) / v_sn_per_atom) * 100
            
            # Theoretical density
            mass_sn = 118.71  # g/mol
            mass_li = 6.94    # g/mol
            mass_li2sn5 = 2 * mass_li + 5 * mass_sn
            
            density_sn = (mass_sn / (v_sn_per_sn * 1e-24 * 6.022e23))
            density_li2sn5 = (mass_li2sn5 / (v_li2sn5_total * 1e-24 * 6.022e23)) * (num_sn_li2sn5 / 5)
            
            st.success("### Volume Expansion Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Volume Expansion per Sn",
                    f"{expansion_per_sn:.2f}%",
                    delta=f"{expansion_per_sn:.2f}%"
                )
                st.write(f"BCT Sn: {v_sn_per_sn:.4f} Å³/Sn")
                st.write(f"Li₂Sn₅: {v_li2sn5_per_sn:.4f} Å³/Sn")
                
            with col2:
                st.metric(
                    "Volume Expansion per atom",
                    f"{expansion_per_atom:.2f}%",
                    delta=f"{expansion_per_atom:.2f}%"
                )
                st.write(f"BCT Sn: {v_sn_per_atom:.4f} Å³/atom")
                st.write(f"Li₂Sn₅: {v_li2sn5_per_atom:.4f} Å³/atom")
            
            st.subheader("Material Properties")
            st.write(f"**Theoretical density of BCT Sn:** {density_sn:.4f} g/cm³")
            st.write(f"**Theoretical density of Li₂Sn₅:** {density_li2sn5:.4f} g/cm³")
            st.write(f"**Density change:** {(density_li2sn5 - density_sn)/density_sn*100:+.2f}%")
            
            return True
            
        except Exception as e:
            st.error(f"Volume expansion calculation failed: {str(e)}")
            st.code(traceback.format_exc())
            return False

run_expansion = st.button("Compute Comprehensive Volume Expansion")

if run_expansion:
    st.info("Starting comprehensive volume expansion analysis...")
    success = calculate_volume_expansion()
    if success:
        st.balloons()
