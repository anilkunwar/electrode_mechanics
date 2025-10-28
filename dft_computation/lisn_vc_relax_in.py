import streamlit as st
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.optimize import BFGS
import os

st.title("ASE + Quantum ESPRESSO: vc-relax for Li and Sn")

st.markdown("""
This app performs a variable-cell relaxation (vc-relax) for Lithium and Tin crystals using ASE + Quantum ESPRESSO.
""")

# Select element
element = st.selectbox("Select Element", options=['Li', 'Sn'])

# Initial lattice constant in Bohr
alat = st.number_input("Initial lattice constant (alat, Bohr)", min_value=2.0, max_value=10.0, value=5.0 if element == 'Li' else 7.5)

# Convergence thresholds and cutoffs
etot_conv_thr = st.number_input("Total energy convergence threshold (Ry)", min_value=1e-8, max_value=1e-4, value=1e-6, format="%.1e")
forc_conv_thr = st.number_input("Force convergence threshold (Ry/Bohr)", min_value=1e-8, max_value=1e-4, value=1e-5, format="%.1e")
ecutwfc = st.number_input("Wavefunction cutoff (Ry)", min_value=20, max_value=100, value=50)
ecutrho = st.number_input("Charge density cutoff (Ry)", min_value=100, max_value=600, value=400 if element == 'Li' else 500)
kpts = st.number_input("K-points grid", min_value=1, max_value=20, value=10 if element == 'Li' else 8)

# Paths and pseudopotentials
pw_command = st.text_input("Path to pw.x executable", value="/usr/local/bin/pw.x")

pseudo_dir = st.text_input("Path to pseudopotential directory", value="./pseudo_dir")
pseudo_dir_abs = os.path.abspath(pseudo_dir)

# Define pseudopotentials for Li and Sn (update filenames as appropriate)
pseudopotentials_map = {
    'Li': 'Li.pbe-s-kjpaw_psl.1.0.0.UPF',
    'Sn': 'Sn.pbe-dn-kjpaw_psl.1.0.0.UPF'
}
pseudo_file = pseudopotentials_map[element]

run_calc = st.button("Run vc-relax Calculation")

if run_calc:
    pw_command_abs = os.path.abspath(pw_command)
    pseudo_path = os.path.join(pseudo_dir_abs, pseudo_file)

    errors = []
    if not (os.path.isfile(pw_command_abs) and os.access(pw_command_abs, os.X_OK)):
        errors.append(f"pw.x executable not found or not executable: {pw_command_abs}")
    if not os.path.isfile(pseudo_path):
        errors.append(f"Pseudopotential file does not exist: {pseudo_path}")

    if errors:
        for err in errors:
            st.error(err)
    else:
        # Convert lattice constant to Angstrom for ASE
        alat_angstrom = alat * 0.529177

        # Setup the atom with cubic cell of size alat_angstrom
        atoms = Atoms(element,
                      positions=[(0, 0, 0)],
                      cell=[(alat_angstrom, 0, 0),
                            (0, alat_angstrom, 0),
                            (0, 0, alat_angstrom)],
                      pbc=True)

        profile = EspressoProfile(
            command=pw_command_abs,
            pseudo_dir=pseudo_dir_abs
        )

        input_data = {
            'control': {
                'calculation': 'vc-relax',
                'prefix': element.lower(),
                'outdir': '/tmp/',
                'pseudo_dir': pseudo_dir_abs,
                'etot_conv_thr': etot_conv_thr,
                'forc_conv_thr': forc_conv_thr
            },
            'system': {
                'nat': 1,
                'ntyp': 1,
                'ecutwfc': ecutwfc,
                'ecutrho': ecutrho,
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.01
            },
            'electrons': {
                'conv_thr': 1e-8
            },
            'ions': {
            },
            'cell': {
                'cell_dofree': 'ibrav'
            }
        }

        calc = Espresso(profile=profile,
                        pseudopotentials={element: pseudo_file},
                        input_data=input_data,
                        kpts=(kpts, kpts, kpts))

        atoms.calc = calc

        try:
            st.write(f"Starting variable-cell relaxation for {element}...")
            opt = BFGS(atoms)
            opt.run(fmax=forc_conv_thr)

            final_cell = atoms.get_cell()
            lattice_constants_bohr = [length / 0.529177 for length in final_cell.lengths()]

            st.success(f"Calculation completed! Optimized lattice constants (Bohr): {lattice_constants_bohr}")
        except Exception as e:
            st.error(f"Calculation failed: {e}")
