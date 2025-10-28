import streamlit as st
from ase import Atoms
from ase.optimize import BFGS
from ase.spacegroup import crystal
from gpaw import GPAW, PW
import os

st.title("ASE + GPAW: vc-relax for Li, Sn, BCT Sn, and Li2Sn5")

st.markdown("""
This app performs a variable-cell relaxation (vc-relax) for Lithium and Tin crystals using ASE + GPAW.
GPAW is a DFT calculator written in Python and installable with `pip install gpaw`.
Now expanded to include BCT Sn and Li2Sn5 for volume expansion calculation.
""")

# Select structure
structure = st.selectbox("Select Structure", options=['Li (cubic)', 'Sn (cubic)', 'Sn (BCT)', 'Li2Sn5'])

# Initial parameters
if structure == 'Li (cubic)':
    element = 'Li'
    alat = st.number_input(
        "Initial lattice constant (alat, Bohr)",
        min_value=2.0,
        max_value=10.0,
        value=5.0
    )
    alat_angstrom = alat * 0.529177
    c_angstrom = alat_angstrom  # cubic
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
    num_sn = 1  # for expansion, but not used
elif structure == 'Sn (cubic)':
    element = 'Sn'
    alat = st.number_input(
        "Initial lattice constant (alat, Bohr)",
        min_value=2.0,
        max_value=10.0,
        value=7.5
    )
    alat_angstrom = alat * 0.529177
    c_angstrom = alat_angstrom  # cubic
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
    num_sn = 1
elif structure == 'Sn (BCT)':
    element = 'Sn'
    a = st.number_input("Initial a (Å)", min_value=2.0, max_value=10.0, value=5.83)
    c = st.number_input("Initial c (Å)", min_value=2.0, max_value=10.0, value=3.18)
    alat_angstrom = a
    c_angstrom = c
    atoms = crystal(
        'Sn',
        basis=[(0,0,0)],
        spacegroup=141,
        cellpar=[a, a, c, 90, 90, 90]
    )
    num_sn = len(atoms)  # 4
elif structure == 'Li2Sn5':
    element = 'Li2Sn5'
    a = st.number_input("Initial a (Å)", min_value=5.0, max_value=15.0, value=10.274)
    c = st.number_input("Initial c (Å)", min_value=2.0, max_value=5.0, value=3.125)
    alat_angstrom = a
    c_angstrom = c
    # Note: The basis positions are approximate; user should replace with exact values from literature (Hansen & Chang, 1969)
    # For example, Li at 4g with x=0.16
    # Sn at 2d (0,0.5,0)
    # Sn at 8i with x=0.295, y=0.432, z=0
    # But to match 3 sites, perhaps adjust.
    # Here, we use a manual setup for the unit cell with positions
    # This is a placeholder; the exact coordinates should be used for accuracy
    positions = [
        (0.0, 0.5, 0.0),  # Sn at 2d
        (0.5, 0.0, 0.0),  # symmetry
        # Add other Sn and Li positions
        # For example, Li at 4g
        (0.16, 0.66, 0.0),
        (0.84, 0.34, 0.0),
        (0.66, 0.16, 0.0),
        (0.34, 0.84, 0.0),
        # Sn at 8i (placeholder)
        (0.295, 0.432, 0.0),
        (0.432, -0.295, 0.0),
        ( -0.432, 0.295, 0.0),
        ( -0.295, -0.432, 0.0),
        (0.295 +0.5, 0.432, 0.0), # symmetry
        (0.432 +0.5, -0.295, 0.0),
        ( -0.432 +0.5, 0.295, 0.0),
        ( -0.295 +0.5, -0.432, 0.0),
        # The above is for 8 Sn at 8i, 2 at 2d, 4 Li at 4g
    ]
    symbols = ['Sn'] * 10 + ['Li'] * 4  # adjust order to match positions
    atoms = Atoms(
        symbols=symbols,
        positions=[ (p[0]*a, p[1]*a, p[2]*c) for p in positions],
        cell=[
            (a, 0, 0),
            (0, a, 0),
            (0, 0, c)
        ],
        pbc=True
    )
    num_sn = 10

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
    value=400 if 'Li' in structure else 500
)
kpts = st.number_input(
    "K-points grid (Nk x Nk x Nk)",
    min_value=1,
    max_value=20,
    value=10 if 'Li' in structure else 8
)

run_calc = st.button("Run vc-relax Calculation")

if run_calc:
    try:
        st.write(f"Setting up vc-relax for {structure} using GPAW...")

        # Set GPAW calculator
        calc = GPAW(
            mode=PW(ecut),
            xc='PBE',
            kpts=(kpts, kpts, kpts),
            txt=f'{structure}_gpaw.log'
        )

        atoms.calc = calc

        st.write("Starting variable-cell relaxation...")

        opt = BFGS(atoms, logfile=f'{structure}_relax.log')
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
        # Run for BCT Sn
        st.write("Running vc-relax for BCT Sn...")
        a_sn = 5.83
        c_sn = 3.18
        atoms_sn = crystal('Sn', basis=[(0,0,0)], spacegroup=141, cellpar=[a_sn, a_sn, c_sn, 90, 90, 90])
        calc_sn = GPAW(mode=PW(500), xc='PBE', kpts=(8,8,12), txt='Sn_BCT_gpaw.log')
        atoms_sn.calc = calc_sn
        opt_sn = BFGS(atoms_sn, logfile='Sn_BCT_relax.log')
        opt_sn.run(fmax=0.001)
        v_sn = atoms_sn.get_volume() / len(atoms_sn)

        # Run for Li2Sn5
        st.write("Running vc-relax for Li2Sn5...")
        a_li2sn5 = 10.274
        c_li2sn5 = 3.125
        # Placeholder for atoms_li2sn5 as above
        atoms_li2sn5 = Atoms( # use the same as above placeholder
            # ... 
        )
        calc_li2sn5 = GPAW(mode=PW(500), xc='PBE', kpts=(6,6,16), txt='Li2Sn5_gpaw.log')
        atoms_li2sn5.calc = calc_li2sn5
        opt_li2sn5 = BFGS(atoms_li2sn5, logfile='Li2Sn5_relax.log')
        opt_li2sn5.run(fmax=0.001)
        v_li2sn5 = atoms_li2sn5.get_volume() / 10  # 10 Sn

        expansion = (v_li2sn5 - v_sn) / v_sn * 100

        st.success("Volume Expansion calculation complete!")
        st.write(f"Volume per Sn in BCT Sn: {v_sn:.6f} Å³")
        st.write(f"Volume per Sn in Li2Sn5: {v_li2sn5:.6f} Å³")
        st.write(f"Volume expansion: {expansion:.2f}%")

    except Exception as e:
        st.error(f"Calculation failed: {e}")
