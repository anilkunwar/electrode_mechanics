import streamlit as st
from ase import Atoms
from ase.optimize import BFGS
from ase.spacegroup import crystal
#from ase.constraints import ExpCellFilter
try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter
from gpaw import GPAW, PW
import os

st.title("ASE + GPAW: vc-relax for Li (BCC), Sn (diamond cubic), Sn (BCT), and Li2Sn5")

st.markdown("""
This app performs a variable-cell relaxation (vc-relax) for Lithium and Tin crystals using ASE + GPAW.
GPAW is a DFT calculator written in Python and installable with `pip install gpaw`.
Expanded to include BCT Sn and Li2Sn5 for volume expansion calculation during lithiation.
""")

# Select structure
structure = st.selectbox("Select Structure", options=['Li (BCC)', 'Sn (diamond cubic)', 'Sn (BCT)', 'Li2Sn5'])

# Default k-points based on structure
if structure == 'Li (BCC)':
    default_ka_kb = 10
    default_kc = 10
    default_ecut = 400
elif structure == 'Sn (diamond cubic)':
    default_ka_kb = 8
    default_kc = 8
    default_ecut = 500
elif structure == 'Sn (BCT)':
    default_ka_kb = 8
    default_kc = 12
    default_ecut = 500
elif structure == 'Li2Sn5':
    default_ka_kb = 6
    default_kc = 16
    default_ecut = 500

# Initial parameters
if structure == 'Li (BCC)':
    a = st.number_input("Initial a (Å)", min_value=2.0, max_value=10.0, value=3.49)
    atoms = crystal('Li', basis=[(0,0,0)], spacegroup=229, cellpar=[a, a, a, 90, 90, 90])
    num_sn = 0  # Not used
    is_cubic = True
elif structure == 'Sn (diamond cubic)':
    a = st.number_input("Initial a (Å)", min_value=4.0, max_value=10.0, value=6.49)
    atoms = crystal('Sn', basis=[(0,0,0), (0.25,0.25,0.25)], spacegroup=227, cellpar=[a, a, a, 90, 90, 90])
    num_sn = 8
    is_cubic = True
elif structure == 'Sn (BCT)':
    a = st.number_input("Initial a (Å)", min_value=2.0, max_value=10.0, value=5.83)
    c = st.number_input("Initial c (Å)", min_value=2.0, max_value=10.0, value=3.18)
    atoms = crystal('Sn', basis=[(0,0,0)], spacegroup=141, cellpar=[a, a, c, 90, 90, 90])
    num_sn = len(atoms)  # 4
    is_cubic = False
elif structure == 'Li2Sn5':
    a = st.number_input("Initial a (Å)", min_value=5.0, max_value=15.0, value=10.274)
    c = st.number_input("Initial c (Å)", min_value=2.0, max_value=5.0, value=3.125)
    atoms = crystal(
        symbols=['Sn', 'Li', 'Sn'],
        basis=[(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)],
        spacegroup=127,
        cellpar=[a, a, c, 90, 90, 90]
    )
    num_sn = 10  # 10 Sn atoms
    is_cubic = False

# Convergence thresholds and cutoffs
etot_conv_thr = st.number_input(
    "SCF energy convergence threshold (eV/atom)",
    min_value=1e-8,
    max_value=1e-3,
    value=1e-5,
    format="%.1e"
)
forc_conv_thr = st.number_input(
    "Force convergence threshold (eV/Å)",
    min_value=1e-8,
    max_value=1e-2,
    value=1e-3,
    format="%.1e"
)
ecut = st.number_input(
    "Plane-wave cutoff energy (eV)",
    min_value=100,
    max_value=1000,
    value=default_ecut
)
ka_kb = st.number_input(
    "K-points along a/b (ka x kb)",
    min_value=1,
    max_value=20,
    value=default_ka_kb
)
kc = st.number_input(
    "K-points along c (kc)",
    min_value=1,
    max_value=20,
    value=default_kc
)

run_calc = st.button("Run vc-relax Calculation")

if run_calc:
    try:
        st.write(f"Setting up vc-relax for {structure} using GPAW...")

        # Set GPAW calculator
        calc = GPAW(
            mode=PW(ecut),
            xc='PBE',
            kpts=(ka_kb, ka_kb, kc),
            convergence={'energy': etot_conv_thr},
            txt=f'{structure}_gpaw.log'
        )

        atoms.calc = calc

        st.write("Starting variable-cell relaxation...")

        ecf = ExpCellFilter(atoms)
        opt = BFGS(ecf, logfile=f'{structure}_relax.log')
        opt.run(fmax=forc_conv_thr)

        final_cell = atoms.get_cell()
        lattice_constants_angstrom = final_cell.lengths()
        total_energy = atoms.get_potential_energy()
        volume = atoms.get_volume()

        st.success(f"Relaxation complete for {structure}!")
        st.write(f"Optimized lattice constants (Å): {lattice_constants_angstrom}")
        st.write(f"Total energy (eV): {total_energy:.6f}")
        st.write(f"Volume (Å³): {volume:.6f}")

    except Exception as e:
        st.error(f"Calculation failed: {e}")

# For volume expansion
run_expansion = st.button("Compute Volume Expansion (BCT Sn to Li2Sn5)")

if run_expansion:
    try:
        # Default parameters for consistency
        etot_conv_thr_exp = 1e-5
        forc_conv_thr_exp = 1e-3
        ecut_exp = 500

        # Run for BCT Sn
        st.write("Running vc-relax for BCT Sn...")
        a_sn = 5.83
        c_sn = 3.18
        atoms_sn = crystal('Sn', basis=[(0,0,0)], spacegroup=141, cellpar=[a_sn, a_sn, c_sn, 90, 90, 90])
        calc_sn = GPAW(
            mode=PW(ecut_exp),
            xc='PBE',
            kpts=(8, 8, 12),
            convergence={'energy': etot_conv_thr_exp},
            txt='Sn_BCT_gpaw.log'
        )
        atoms_sn.calc = calc_sn
        ecf_sn = ExpCellFilter(atoms_sn)
        opt_sn = BFGS(ecf_sn, logfile='Sn_BCT_relax.log')
        opt_sn.run(fmax=forc_conv_thr_exp)
        num_sn_atoms = len(atoms_sn)  # 4, all Sn
        v_sn = atoms_sn.get_volume() / num_sn_atoms

        # Run for Li2Sn5
        st.write("Running vc-relax for Li2Sn5...")
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
            kpts=(6, 6, 16),
            convergence={'energy': etot_conv_thr_exp},
            txt='Li2Sn5_gpaw.log'
        )
        atoms_li2sn5.calc = calc_li2sn5
        ecf_li2sn5 = ExpCellFilter(atoms_li2sn5)
        opt_li2sn5 = BFGS(ecf_li2sn5, logfile='Li2Sn5_relax.log')
        opt_li2sn5.run(fmax=forc_conv_thr_exp)
        num_sn_li2sn5 = atoms_li2sn5.get_chemical_symbols().count('Sn')  # 10
        v_li2sn5 = atoms_li2sn5.get_volume() / num_sn_li2sn5

        expansion = (v_li2sn5 - v_sn) / v_sn * 100

        st.success("Volume Expansion calculation complete!")
        st.write(f"Volume per Sn in BCT Sn: {v_sn:.6f} Å³")
        st.write(f"Volume per Sn in Li2Sn5: {v_li2sn5:.6f} Å³")
        st.write(f"Volume expansion: {expansion:.2f}%")

    except Exception as e:
        st.error(f"Calculation failed: {e}")
