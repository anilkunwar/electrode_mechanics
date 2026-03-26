import streamlit as st
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.spacegroup import crystal
#from ase.constraints import ExpCellFilter
try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter
from ase.units import GPa
from ase.eos import EquationOfState
from gpaw import GPAW, PW
import matplotlib.pyplot as plt
import os

st.title("DFT Volume Expansion via E(V) Curve & EOS Fitting (ASE + GPAW)")

st.markdown("""
This app computes **volume expansion** during lithiation (BCT Sn → Li₂Sn₅) using **energy-volume (E-V) curves** and **Birch-Murnaghan Equation of State (EOS)** fitting.

**Scientific Method**:
1. For each phase, scale the unit cell volume (90% to 110% of initial).
2. Relax atomic positions at **fixed volume**.
3. Compute total energy E(V).
4. Fit E(V) to Birch-Murnaghan EOS → extract **equilibrium volume V₀**.
5. Normalize by number of Sn atoms.
6. Compute:  
   \[
   \text{Expansion (\%)} = \left( \frac{V_{\text{Li}_2\text{Sn}_5} - V_{\text{Sn}}}{V_{\text{Sn}}} \right) \times 100
   \]

**Fast & Accurate**: Uses GPAW with PBE, plane-waves, and ASE's EOS module.
""")

# Sidebar: Calculation Settings
st.sidebar.header("DFT & EOS Settings")

calculation_mode = st.sidebar.selectbox(
    "Calculation Mode",
    ["Fast Testing", "Balanced", "High Accuracy"],
    index=0
)

if calculation_mode == "Fast Testing":
    ecut = 350
    kpts_sn = (4, 4, 6)
    kpts_li2sn5 = (3, 3, 8)
    fmax = 0.05
    volume_points = 7
elif calculation_mode == "Balanced":
    ecut = 450
    kpts_sn = (6, 6, 10)
    kpts_li2sn5 = (4, 4, 12)
    fmax = 0.01
    volume_points = 9
else:  # High Accuracy
    ecut = 500
    kpts_sn = (8, 8, 12)
    kpts_li2sn5 = (6, 6, 16)
    fmax = 0.005
    volume_points = 11

volume_range = st.sidebar.slider("Volume Scaling Range", 0.8, 1.2, (0.9, 1.1))
n_points = st.sidebar.number_input("Number of Volume Points", min_value=5, max_value=15, value=volume_points)

# Function to compute E(V) for a structure
@st.cache_data(show_spinner=False)
def compute_ev_curve(structure_name, a_init, c_init, basis, spacegroup, wyckoff_symbols, wyckoff_basis, num_sn, kpts):
    if structure_name == 'Sn (BCT)':
        atoms_template = crystal('Sn', basis=[(0,0,0)], spacegroup=141, cellpar=[a_init, a_init, c_init, 90, 90, 90])
    elif structure_name == 'Li2Sn5':
        atoms_template = crystal(
            symbols=wyckoff_symbols,
            basis=wyckoff_basis,
            spacegroup=127,
            cellpar=[a_init, a_init, c_init, 90, 90, 90]
        )
    else:
        raise ValueError("Unknown structure")

    v0 = atoms_template.get_volume()
    volumes_rel = np.linspace(volume_range[0], volume_range[1], n_points)
    volumes = v0 * volumes_rel
    energies = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, v in enumerate(volumes):
        status_text.text(f"Computing {structure_name}: Volume {i+1}/{n_points} ({v/v0*100:.1f}% of V₀)")
        
        # Scale cell preserving shape
        scale = (v / v0) ** (1.0 / 3.0)
        cell = atoms_template.get_cell() * scale
        atoms = atoms_template.copy()
        atoms.set_cell(cell, scale_atoms=True)

        # GPAW calculator
        calc = GPAW(
            mode=PW(ecut),
            xc='PBE',
            kpts=kpts,
            txt=f'{structure_name}_ev.log',
            convergence={'energy': 1e-5},
            maxiter=200
        )
        atoms.calc = calc

        # Relax ions at fixed volume
        opt = BFGS(atoms, logfile=f'{structure_name}_relax.log')
        opt.run(fmax=fmax)

        energy = atoms.get_potential_energy()
        energies.append(energy)
        progress_bar.progress((i + 1) / n_points)

    status_text.text(f"{structure_name}: E(V) curve complete!")
    progress_bar.empty()

    return volumes, np.array(energies), v0, num_sn

# Main calculation
run_calc = st.button("Run E(V) Curve & Volume Expansion")

if run_calc:
    try:
        with st.spinner("Computing BCT Sn E(V) curve..."):
            v_sn_list, e_sn_list, v0_sn, num_sn_sn = compute_ev_curve(
                'Sn (BCT)', 5.83, 3.18, None, 141, None, None, 4, kpts_sn
            )
        
        with st.spinner("Computing Li₂Sn₅ E(V) curve..."):
            v_li2sn5_list, e_li2sn5_list, v0_li2sn5, num_sn_li2sn5 = compute_ev_curve(
                'Li2Sn5', 10.274, 3.125, ['Sn', 'Li', 'Sn'],
                127, [(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)], 10, kpts_li2sn5
            )

        # Fit EOS
        eos_sn = EquationOfState(v_sn_list, e_sn_list)
        v0_fit_sn, e0_sn, B_sn, Bp_sn = eos_sn.fit()
        v_per_sn_sn = v0_fit_sn / num_sn_sn

        eos_li2sn5 = EquationOfState(v_li2sn5_list, e_li2sn5_list)
        v0_fit_li2sn5, e0_li2sn5, B_li2sn5, Bp_li2sn5 = eos_li2sn5.fit()
        v_per_sn_li2sn5 = v0_fit_li2sn5 / num_sn_li2sn5

        # Volume expansion
        expansion = (v_per_sn_li2sn5 - v_per_sn_sn) / v_per_sn_sn * 100

        # Results
        st.success("E(V) Fitting & Volume Expansion Complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("BCT Sn")
            st.metric("V₀ (fit)", f"{v0_fit_sn:.3f} Å³")
            st.metric("V/Sn", f"{v_per_sn_sn:.3f} Å³")
            st.metric("Bulk Modulus", f"{B_sn/GPa:.1f} GPa")

        with col2:
            st.subheader("Li₂Sn₅")
            st.metric("V₀ (fit)", f"{v0_fit_li2sn5:.3f} Å³")
            st.metric("V/Sn", f"{v_per_sn_li2sn5:.3f} Å³")
            st.metric("Bulk Modulus", f"{B_li2sn5/GPa:.1f} GPa")

        st.metric("Volume Expansion (BCT Sn → Li₂Sn₅)", f"{expansion:.2f}%", delta=f"+{expansion:.2f}%")

        # Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        eos_sn.plot(ax1)
        ax1.set_title("BCT Sn: E(V) Curve & EOS Fit")

        eos_li2sn5.plot(ax2)
        ax2.set_title("Li₂Sn₅: E(V) Curve & EOS Fit")

        st.pyplot(fig)

        # Download data
        import pandas as pd
        df_sn = pd.DataFrame({'Volume (Å³)': v_sn_list, 'Energy (eV)': e_sn_list})
        df_li2sn5 = pd.DataFrame({'Volume (Å³)': v_li2sn5_list, 'Energy (eV)': e_li2sn5_list})

        st.download_button("Download BCT Sn E(V) Data", df_sn.to_csv(index=False), "sn_ev.csv")
        st.download_button("Download Li₂Sn₅ E(V) Data", df_li2sn5.to_csv(index=False), "li2sn5_ev.csv")

    except Exception as e:
        st.error(f"Calculation failed: {e}")
        st.error("Check logs in working directory.")

# Tips
with st.expander("Tips for Accuracy & Speed"):
    st.markdown("""
    - **Fast Testing**: Quick trends (~5–15 min per phase).
    - **Balanced**: Good for papers (~30–90 min).
    - **High Accuracy**: Publication-ready (~2–6 hrs).
    - Use **cluster** for large cells or high k-point density.
    - PBE typically **overestimates volume by 1–3%** vs. experiment.
    """)
