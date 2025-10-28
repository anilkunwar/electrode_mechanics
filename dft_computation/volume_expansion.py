import streamlit as st
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from gpaw import GPAW, PW
import sqlite3
import pandas as pd
from ase.cell import Cell  # Fixed import for cell setup

# Page Setup
st.set_page_config(page_title="ASE + GPAW vc-relax for Li-Sn", layout="centered")
st.title("ASE + GPAW: vc-relax for Li-Sn Phases & Volume Expansion")

st.markdown("""
Perform vc-relax for pure BCT Sn and LiSn. Calculate volume expansion of LiSn vs. BCT Sn.
Note: Use Fermi smearing for metals to avoid convergence errors. Calculations may take minutes.
""")

# Database Setup
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
            ecut REAL,
            kpts_x INTEGER,
            kpts_y INTEGER,
            kpts_z INTEGER,
            energy REAL,
            volume REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_result(phase, num_li, num_sn, ecut, kpts_x, kpts_y, kpts_z, energy, volume):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO relax_results 
        (phase, num_li, num_sn, ecut, kpts_x, kpts_y, kpts_z, energy, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (phase, num_li, num_sn, ecut, kpts_x, kpts_y, kpts_z, energy, volume))
    conn.commit()
    conn.close()

# Reusable vc-relax Function
def run_vc_relax(phase, default_ecut, default_kpts, initial_atoms_factory):
    st.subheader(f"{phase} Relaxation Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ecut = st.number_input(
            "Plane-wave cutoff energy (eV)",
            min_value=100, max_value=1000, value=default_ecut, 
            key=f"ecut_{phase}"
        )
        
        kpts_x = st.number_input(
            "K-points x",
            min_value=1, max_value=20, value=default_kpts[0],
            key=f"kpts_x_{phase}"
        )
        
        kpts_y = st.number_input(
            "K-points y",
            min_value=1, max_value=20, value=default_kpts[1],
            key=f"kpts_y_{phase}"
        )
        
        kpts_z = st.number_input(
            "K-points z",
            min_value=1, max_value=20, value=default_kpts[2],
            key=f"kpts_z_{phase}"
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
        with st.spinner(f"Running vc-relax for {phase}..."):
            try:
                atoms = initial_atoms_factory()

                st.write("### Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()

                # GPAW with smearing to fix errors
                status_text.text("Initializing GPAW...")
                calc = GPAW(
                    mode=PW(ecut),
                    xc='PBE',
                    kpts=(kpts_x, kpts_y, kpts_z),
                    occupations={'name': 'fermi', 'width': 0.1},  # Fix for positive definite error
                    convergence={'energy': etot_conv_thr},
                    txt=None,
                    mixer={'backend': 'pulay', 'beta': 0.1, 'nmaxold': 5, 'weight': 100}
                )

                atoms.calc = calc

                status_text.text("Starting vc-relax...")
                ucf = UnitCellFilter(atoms)
                opt = BFGS(ucf, logfile=None)
                
                for i in range(max_steps):
                    status_text.text(f"Step {i+1}/{max_steps}")
                    progress_bar.progress((i + 1) / max_steps)
                    opt.run(fmax=forc_conv_thr, steps=1)
                    if opt.converged():
                        break

                # Results
                final_energy = atoms.get_potential_energy()
                final_volume = atoms.get_volume() / (0.529177 ** 3)  # Bohr³
                num_li = sum(1 for a in atoms if a.symbol == 'Li')
                num_sn = sum(1 for a in atoms if a.symbol == 'Sn')

                st.success("✅ Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Energy", f"{final_energy:.6f} eV")
                with col2:
                    st.metric("Final Volume", f"{final_volume:.4f} Bohr³")
                
                # Save
                save_result(
                    phase=phase,
                    num_li=num_li,
                    num_sn=num_sn,
                    ecut=ecut,
                    kpts_x=kpts_x,
                    kpts_y=kpts_y,
                    kpts_z=kpts_z,
                    energy=final_energy,
                    volume=final_volume
                )
                st.info("📊 Saved to database")

            except Exception as e:
                st.error(f"❌ Failed: {str(e)}")
                st.info("Try higher ecut (600+), adjust kpts, or check structure.")

# Initial Structures
def create_bct_sn():
    a = 5.831  # Å
    c = 3.182  # Å
    atoms = Atoms('Sn2',
                  scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
                  cell=[[a, 0, 0], [0, a, 0], [0, 0, c]],
                  pbc=True)
    return atoms

def create_lisn():
    # Approximate from literature/MP (monoclinic C2/c, Z=4); replace with ase.io.read('LiSn.cif') for exact
    a = 5.335  # Å
    b = 9.213
    c = 5.228
    alpha = 90
    beta = 112.44
    gamma = 90
    cell = Cell.fromcellpar([a, b, c, alpha, beta, gamma])
    # Fractional positions (approximate; Li on 4e, Sn on 4f)
    scaled_positions = [
        (0, 0.2581, 0.25),  # Li1
        (0, 0.7419, 0.75),  # Li1 symmetry
        (0.5, 0.7581, 0.25),  # Li1
        (0.5, 0.2419, 0.75),  # Li1
        (0.25, 0.0838, 0),  # Sn1
        (0.75, 0.0838, 0.5),  # Sn1
        (0.75, -0.0838, 0),  # Sn1
        (0.25, -0.0838, 0.5)  # Sn1
    ]
    symbols = ['Li'] * 4 + ['Sn'] * 4
    atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)
    return atoms

# Tabs
tab_sn, tab_lisn, tab_db, tab_info = st.tabs(["Pure BCT Sn", "LiSn", "Results & Expansion", "Instructions"])

with tab_sn:
    st.info("BCT Sn: ecut ~500 eV, kpts (6,6,11) for aspect ratio")
    run_vc_relax(phase='BCT_Sn', default_ecut=500, default_kpts=(6,6,11), initial_atoms_factory=create_bct_sn)

with tab_lisn:
    st.info("LiSn (monoclinic C2/c): ecut ~500 eV, kpts (6,4,6). Positions approximate; use CIF for exact.")
    run_vc_relax(phase='LiSn', default_ecut=500, default_kpts=(6,4,6), initial_atoms_factory=create_lisn)

with tab_db:
    st.subheader("Results")
    
    if st.button("Clear Database"):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM relax_results")
        conn.commit()
        conn.close()
        st.success("Cleared!")
        st.rerun()
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM relax_results ORDER BY timestamp DESC", conn)
    conn.close()
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Volume Expansion")
        if 'BCT_Sn' in df['phase'].values and 'LiSn' in df['phase'].values:
            sn_row = df[df['phase'] == 'BCT_Sn'].iloc[0]
            lisn_row = df[df['phase'] == 'LiSn'].iloc[0]
            v_sn_per_sn = sn_row['volume'] / sn_row['num_sn']
            v_lisn_per_sn = lisn_row['volume'] / lisn_row['num_sn']
            expansion = ((v_lisn_per_sn - v_sn_per_sn) / v_sn_per_sn) * 100
            st.metric("Expansion (%) for LiSn vs. BCT Sn (per Sn)", f"{expansion:.2f}%")
        else:
            st.info("Run both phases to compute.")
    else:
        st.info("No results. Run calculations.")

with tab_info:
    st.markdown("""
    ### Parameters
    - BCT Sn: a≈5.831 Å, c≈3.182 Å; adjust kpts for ratio.
    - LiSn: a≈5.335 Å, b≈9.213 Å, c≈5.228 Å, β≈112.44°; positions approximate.
    - Smearing added for convergence; increase ecut if errors persist.
    - For exact LiSn, download CIF from Materials Project mp-13444 and replace factory with ase.io.read.
    - Expansion: ((V_LiSn / num_Sn_LiSn - V_Sn / num_Sn_Sn) / (V_Sn / num_Sn_Sn)) * 100%
    """)

st.markdown("---")
st.caption("Improved ASE + GPAW App | Made with Streamlit")
