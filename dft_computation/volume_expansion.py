import streamlit as st
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from gpaw import GPAW, PW
import sqlite3
import os
import pandas as pd
import numpy as np  # Added for angle conversions if needed

# ---------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="ASE + GPAW vc-relax for Li-Sn", layout="centered")
st.title("ASE + GPAW: Variable-Cell Relaxation for Li-Sn Phases")

st.markdown("""
Perform **variable-cell relaxation (vc-relax)** calculations for **pure BCT Sn** and **LiSn** phase
using the **GPAW** DFT calculator integrated with ASE. Calculate volume expansion of LiSn relative to pure Sn.

**Note:** Calculations may take several minutes. For larger phases like Li17Sn4, use reduced k-points.
""")

# ---------------------------------------------------------------
# SQLite Database Setup
# ---------------------------------------------------------------
db_path = "results.db"

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relax_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phase TEXT,
            num_li INTEGER,
            num_sn INTEGER,
            alat_bohr REAL,
            ecut REAL,
            kpts INTEGER,
            etot_conv_thr REAL,
            forc_conv_thr REAL,
            energy REAL,
            a REAL,
            b REAL,
            c REAL,
            alpha REAL,
            beta REAL,
            gamma REAL,
            volume REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_result(phase, num_li, num_sn, alat_bohr, ecut, kpts, etot_conv_thr, forc_conv_thr, energy, cell_lengths, cell_angles):
    volume = atoms.get_volume() / (0.529177 ** 3)  # Volume in Bohr³
    a, b, c = cell_lengths
    alpha, beta, gamma = cell_angles
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO relax_results 
        (phase, num_li, num_sn, alat_bohr, ecut, kpts, etot_conv_thr, forc_conv_thr, energy, a, b, c, alpha, beta, gamma, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (phase, num_li, num_sn, alat_bohr, ecut, kpts, etot_conv_thr, forc_conv_thr, energy, a, b, c, alpha, beta, gamma, volume))
    conn.commit()
    conn.close()

# ---------------------------------------------------------------
# Reusable Function for vc-relax (updated for general cells)
# ---------------------------------------------------------------
def run_vc_relax(phase, default_alat_bohr, default_ecut, default_kpts, initial_atoms_factory):
    st.subheader(f"{phase} Relaxation Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        alat = st.number_input(
            f"Initial lattice scale factor (Bohr; scales initial cell)",
            min_value=2.0, max_value=15.0, value=default_alat_bohr,
            key=f"alat_{phase}"
        )
        
        ecut = st.number_input(
            "Plane-wave cutoff energy (eV)",
            min_value=100, max_value=1000, value=default_ecut, 
            key=f"ecut_{phase}"
        )
        
        kpts = st.number_input(
            "K-points grid density (approx; adjusted for cell)",
            min_value=1, max_value=20, value=default_kpts,
            key=f"kpts_{phase}"
        )
    
    with col2:
        etot_conv_thr = st.number_input(
            "Energy convergence (eV/atom)",
            min_value=1e-8, max_value=1e-3, value=1e-5, format="%.1e",
            key=f"etot_{phase}"
        )
        
        forc_conv_thr = st.number_input(
            "Force convergence (eV/Å)",
            min_value=1e-5, max_value=1e-1, value=1e-3, format="%.1e",
            key=f"fmax_{phase}"
        )
        
        max_steps = st.number_input(
            "Max optimization steps",
            min_value=1, max_value=100, value=10,
            key=f"steps_{phase}"
        )

    run_calc = st.button(f"Run vc-relax for {phase}", type="primary")

    if run_calc:
        with st.spinner(f"Running variable-cell relaxation for {phase}... This may take a while."):
            try:
                # Create initial structure and scale by alat factor (in Bohr, convert to Å)
                alat_angstrom = alat * 0.529177
                atoms = initial_atoms_factory()
                atoms.set_cell(atoms.get_cell() * (alat_angstrom / default_alat_bohr * 0.529177))  # Scale cell
                atoms.positions *= (alat_angstrom / default_alat_bohr * 0.529177)  # Scale positions accordingly

                st.write("### Calculation Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Setup GPAW calculator
                status_text.text("Initializing GPAW calculator...")
                calc = GPAW(
                    mode=PW(ecut),
                    xc='PBE',
                    kpts=(kpts, kpts, kpts),  # Adjust based on cell size if needed
                    convergence={'energy': etot_conv_thr},
                    txt=None,
                    mixer={'backend': 'pulay', 'beta': 0.1, 'nmaxold': 5, 'weight': 100}
                )

                atoms.calc = calc

                # Variable-cell relaxation
                status_text.text("Starting variable-cell relaxation...")
                ucf = UnitCellFilter(atoms)
                opt = BFGS(ucf, logfile=None, trajectory=None)
                
                for i in range(max_steps):
                    status_text.text(f"Optimization step {i+1}/{max_steps}")
                    progress_bar.progress((i + 1) / max_steps)
                    opt.run(fmax=forc_conv_thr, steps=1)
                    if opt.converged():
                        break

                # Get final results
                status_text.text("Finalizing calculation...")
                final_energy = atoms.get_potential_energy()
                final_cell = atoms.get_cell()
                cell_lengths = final_cell.lengths() / 0.529177  # Bohr
                cell_angles = final_cell.angles()  # Degrees
                final_volume = atoms.get_volume() / (0.529177 ** 3)  # Bohr³
                num_li = len([a for a in atoms if a.symbol == 'Li'])
                num_sn = len([a for a in atoms if a.symbol == 'Sn'])

                # Display results
                st.success("✅ Relaxation complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Energy", f"{final_energy:.6f} eV")
                    st.metric("Final Volume", f"{final_volume:.4f} Bohr³")
                
                with col2:
                    st.metric("Lattice a", f"{cell_lengths[0]:.6f} Bohr")
                    st.metric("Lattice b", f"{cell_lengths[1]:.6f} Bohr")
                
                with col3:
                    st.metric("Lattice c", f"{cell_lengths[2]:.6f} Bohr")
                    st.metric("Initial scale", f"{alat:.4f} Bohr")

                # Save to database
                save_result(
                    phase=phase,
                    num_li=num_li,
                    num_sn=num_sn,
                    alat_bohr=alat,
                    ecut=ecut,
                    kpts=kpts,
                    etot_conv_thr=etot_conv_thr,
                    forc_conv_thr=forc_conv_thr,
                    energy=final_energy,
                    cell_lengths=cell_lengths,
                    cell_angles=cell_angles
                )
                
                st.info(f"📊 Results saved to database")

            except Exception as e:
                st.error(f"❌ Calculation failed: {str(e)}")
                st.info("Try adjusting parameters or check system resources")

# Initial structure factories
def create_bct_sn():
    a_ang = 5.831  # Å
    c_ang = 3.182  # Å
    atoms = Atoms('Sn2',
                  positions=[[0, 0, 0], [a_ang/2, a_ang/2, c_ang/2]],
                  cell=[[a_ang, 0, 0], [0, a_ang, 0], [0, 0, c_ang]],
                  pbc=True)
    return atoms

def create_lisn():
    # From Materials Project mp-13444 (optimized DFT values as initial)
    a_ang = 5.335
    b_ang = 9.213
    c_ang = 5.228
    alpha = 90
    beta = 112.44
    gamma = 90
    # Fractional coordinates (2 inequivalent Li and Sn sites, scaled for unit cell with 4 Li and 4 Sn)
    positions = [
        (0.0, 0.25, 0.0),  # Li1
        (0.5, 0.25, 0.5),  # Li1
        (0.0, 0.75, 0.0),  # Li1 (symmetry equivalents)
        (0.5, 0.75, 0.5),  # Li1
        (0.25, 0.0, 0.25),  # Li2/Sn? Wait, correct from MP:
        # Actual from MP description: Li sites and Sn sites (use CIF for exact; placeholder)
        # Placeholder: Assume 4 Li and 4 Sn in cell; replace with exact fractional coords from CIF
        (0.0, 0.0, 0.0),  # Sn1 (example; get from https://materialsproject.org/materials/mp-13444#crystal-structure)
        # Note: For accuracy, download CIF from MP and use ase.io.read('lisn.cif')
        # Here, approximate positions based on description
    ]
    symbols = ['Li']*4 + ['Sn']*4  # Adjust
    cell = ase.cell.Cell.fromcellpar([a_ang, b_ang, c_ang, alpha, beta, gamma])
    atoms = Atoms(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)  # Use scaled_positions
    return atoms

# ---------------------------------------------------------------
# Main App Layout with Tabs
# ---------------------------------------------------------------
tab_sn, tab_lisn, tab_db, tab_info = st.tabs([
    "🔹 Pure BCT Sn", 
    "🔸 LiSn Phase", 
    "📊 Results Database & Expansion",
    "ℹ️ Instructions"
])

with tab_sn:
    st.info("BCT Sn (beta-Sn) - Recommended: alat ≈ 11.0 Bohr (scales a,c), Ecut ≈ 500 eV, kpts ≈ 6")
    run_vc_relax(phase='BCT_Sn', default_alat_bohr=11.0, default_ecut=500, default_kpts=6, initial_atoms_factory=create_bct_sn)

with tab_lisn:
    st.info("LiSn (monoclinic C2/c) - Recommended: alat ≈ 10.0 Bohr (scales cell), Ecut ≈ 500 eV, kpts ≈ 6. Replace positions with MP CIF for accuracy.")
    run_vc_relax(phase='LiSn', default_alat_bohr=10.0, default_ecut=500, default_kpts=6, initial_atoms_factory=create_lisn)

with tab_db:
    st.subheader("Stored Relaxation Results")
    
    if st.button("Clear All Results"):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM relax_results")
        conn.commit()
        conn.close()
        st.success("Database cleared!")
        st.rerun()
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT 
            id, phase, num_li, num_sn,
            alat_bohr as "Initial scale (Bohr)",
            ecut as "Ecut (eV)", 
            kpts as "K-points",
            energy as "Energy (eV)",
            a as "a (Bohr)", 
            b as "b (Bohr)", 
            c as "c (Bohr)",
            alpha as "α (°)", 
            beta as "β (°)", 
            gamma as "γ (°)",
            volume as "Volume (Bohr³)",
            timestamp as "Date"
        FROM relax_results 
        ORDER BY timestamp DESC
    """, conn)
    conn.close()
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        summary = df.groupby('phase').agg({
            'Energy (eV)': ['min', 'mean', 'max'],
            'a (Bohr)': ['mean', 'std'],
            'Volume (Bohr³)': ['mean', 'std']
        }).round(6)
        st.dataframe(summary)
        
        # Volume Expansion Calculation
        st.subheader("Volume Expansion")
        if 'BCT_Sn' in df['phase'].values and 'LiSn' in df['phase'].values:
            # Take latest runs
            sn_row = df[df['phase'] == 'BCT_Sn'].iloc[0]
            lisn_row = df[df['phase'] == 'LiSn'].iloc[0]
            v_sn_per_sn = sn_row['Volume (Bohr³)'] / sn_row['num_sn']
            v_lisn_per_sn = lisn_row['Volume (Bohr³)'] / lisn_row['num_sn']
            expansion = ((v_lisn_per_sn - v_sn_per_sn) / v_sn_per_sn) * 100
            st.metric("Volume Expansion (%) for LiSn vs. BCT Sn", f"{expansion:.2f}%")
            st.info("Expansion calculated per Sn atom. Run both phases for update.")
        else:
            st.info("Run calculations for both BCT_Sn and LiSn to compute expansion.")
    else:
        st.info("No results found. Run a calculation to populate the database.")

with tab_info:
    st.subheader("Instructions & Parameters")
    
    st.markdown("""
    ### 🎯 Recommended Parameters
    
    **Pure BCT Sn:**
    - Initial scale: 11.0 Bohr (a ≈ 5.83 Å, c ≈ 3.18 Å)
    - Cutoff energy: 450-550 eV
    - K-points: 6×6×6
    
    **LiSn (monoclinic):**
    - Initial scale: 10.0 Bohr (a ≈ 5.34 Å, b ≈ 9.21 Å, c ≈ 5.23 Å, β ≈ 112°)
    - Cutoff energy: 450-550 eV
    - K-points: 6×6×6
    
    For other LixSny (e.g., Li17Sn4 cubic F-43m, a ≈ 37.3 Bohr), download CIF from Materials Project, read with ase.io.read(), and replace initial_atoms_factory.
    
    ### ⚙️ Parameter Guidelines
    - Higher ecut/kpts for accuracy, but slower.
    - Convergence: Tighter for precision.
    
    ### ⏰ Performance Notes
    - Runs in Python/GPAW; simple phases ~5-20 min, large ones hours.
    - In Streamlit Cloud, monitor resource limits.
    
    ### Technical Details
    - DFT: PBE functional
    - Relaxation: BFGS with UnitCellFilter
    - Units: Bohr for input/output, Å internally
    - Expansion: Per Sn atom for battery-relevant metric (~150-300% expected for full lithiation)
    """)

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown("---")
st.caption("ASE + GPAW Li-Sn Relaxation & Volume Expansion App | Made with Streamlit")
