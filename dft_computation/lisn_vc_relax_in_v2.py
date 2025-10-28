import streamlit as st
from ase import Atoms
from ase.optimize import BFGS
from gpaw import GPAW, PW
import os

st.title("ASE + GPAW: vc-relax for Li and Sn")

st.markdown("""
This app performs a variable-cell relaxation (vc-relax) for Lithium and Tin crystals using ASE + GPAW.
GPAW is a DFT calculator written in Python and installable with `pip install gpaw`.
""")

# Select element
element = st.selectbox("Select Element", options=['Li', 'Sn'])

# Initial lattice constant in Bohr
alat = st.number_input(
    "Initial lattice constant (alat, Bohr)",
    min_value=2.0,
    max_value=10.0,
    value=5.0 if element == 'Li' else 7.5
)

# Convergence thresholds and cutoffs
etot_conv_thr = st.number_input(
    "Total energy convergence threshold (eV)",
    min_value=1e-8,
    max_value=1e-4,
    value=1e-6,
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
    value=400 if element == 'Li' else 500
)
kpts = st.number_input(
    "K-points grid (Nk x Nk x Nk)",
    min_value=1,
    max_value=20,
    value=8 if element == 'Sn' else 10
)

run_calc = st.button("Run vc-relax Calculation")

if run_calc:
    try:
        st.write(f"Setting up vc-relax for {element} using GPAW...")

        # Convert lattice constant to Å
        alat_angstrom = alat * 0.529177

        # Create a simple cubic cell
        atoms = Atoms(
            element,
            positions=[(0, 0, 0)],
            cell=[
                (alat_angstrom, 0, 0),
                (0, alat_angstrom, 0),
                (0, 0, alat_angstrom)
            ],
            pbc=True
        )

        # Set GPAW calculator
        calc = GPAW(
            mode=PW(ecut),
            xc='PBE',
            kpts=(kpts, kpts, kpts),
            txt=f'{element}_gpaw.log'
        )

        atoms.calc = calc

        st.write("Starting variable-cell relaxation...")

        opt = BFGS(atoms, logfile=f'{element}_relax.log')
        opt.run(fmax=forc_conv_thr)

        final_cell = atoms.get_cell()
        lattice_constants_bohr = [length / 0.529177 for length in final_cell.lengths()]
        total_energy = atoms.get_potential_energy()

        st.success(f"Relaxation complete for {element}!")
        st.write(f"Optimized lattice constants (Bohr): {lattice_constants_bohr}")
        st.write(f"Total energy (eV): {total_energy:.6f}")

    except Exception as e:
        st.error(f"Calculation failed: {e}")
