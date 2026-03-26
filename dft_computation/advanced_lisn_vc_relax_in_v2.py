import streamlit as st
from ase import Atoms
from ase.optimize import BFGS
##########################################
# ase.constratints is replaced by ase.filter
#from ase.constraints import UnitCellFilter
try:
    from ase.filters import UnitCellFilter
except ImportError:
    from ase.constraints import UnitCellFilter
#######################333
from gpaw import GPAW, PW
import sqlite3
import os
import pandas as pd

# ---------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="ASE + GPAW vc-relax", layout="centered")
st.title("ASE + GPAW: Variable-Cell Relaxation for Li and Sn")

st.markdown("""
Perform **variable-cell relaxation (vc-relax)** calculations for **Lithium (Li)** and **Tin (Sn)**
using the **GPAW** DFT calculator integrated with ASE.

**Note:** Calculations may take several minutes depending on parameters and system resources.
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
            element TEXT,
            alat_bohr REAL,
            ecut REAL,
            kpts INTEGER,
            etot_conv_thr REAL,
            forc_conv_thr REAL,
            energy REAL,
            a REAL,
            b REAL,
            c REAL,
            volume REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_result(element, alat_bohr, ecut, kpts, etot_conv_thr, forc_conv_thr, energy, a, b, c):
    volume = a * b * c
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO relax_results 
        (element, alat_bohr, ecut, kpts, etot_conv_thr, forc_conv_thr, energy, a, b, c, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (element, alat_bohr, ecut, kpts, etot_conv_thr, forc_conv_thr, energy, a, b, c, volume))
    conn.commit()
    conn.close()

# ---------------------------------------------------------------
# Reusable Function for vc-relax
# ---------------------------------------------------------------
def run_vc_relax(element, default_alat_bohr, default_ecut, default_kpts):
    st.subheader(f"{element} Relaxation Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        alat = st.number_input(
            f"Initial lattice constant (Bohr)",
            min_value=2.0, max_value=15.0, value=default_alat_bohr,
            key=f"alat_{element}"
        )
        
        ecut = st.number_input(
            "Plane-wave cutoff energy (eV)",
            min_value=100, max_value=1000, value=default_ecut, 
            key=f"ecut_{element}"
        )
        
        kpts = st.number_input(
            "K-points grid (Nk x Nk x Nk)",
            min_value=1, max_value=20, value=default_kpts,
            key=f"kpts_{element}"
        )
    
    with col2:
        etot_conv_thr = st.number_input(
            "Energy convergence (eV/atom)",
            min_value=1e-8, max_value=1e-3, value=1e-5, format="%.1e",
            key=f"etot_{element}"
        )
        
        forc_conv_thr = st.number_input(
            "Force convergence (eV/Å)",
            min_value=1e-5, max_value=1e-1, value=1e-3, format="%.1e",
            key=f"fmax_{element}"
        )
        
        max_steps = st.number_input(
            "Max optimization steps",
            min_value=1, max_value=100, value=10,
            key=f"steps_{element}"
        )

    run_calc = st.button(f"Run vc-relax for {element}", type="primary")

    if run_calc:
        with st.spinner(f"Running variable-cell relaxation for {element}... This may take a while."):
            try:
                # Convert Bohr → Å for ASE
                alat_angstrom = alat * 0.529177
                
                st.write("### Calculation Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Create initial BCC structure
                status_text.text("Setting up crystal structure...")
                atoms = Atoms(
                    element,
                    positions=[[0, 0, 0]],
                    cell=[[alat_angstrom, 0, 0],
                         [0, alat_angstrom, 0],
                         [0, 0, alat_angstrom]],
                    pbc=True
                )

                # Setup GPAW calculator
                status_text.text("Initializing GPAW calculator...")
                calc = GPAW(
                    mode=PW(ecut),
                    xc='PBE',
                    kpts=(kpts, kpts, kpts),
                    convergence={'energy': etot_conv_thr},
                    txt=None,  # Disable detailed text output
                    mixer={'backend': 'pulay', 'beta': 0.1, 'nmaxold': 5, 'weight': 100}
                )

                atoms.calc = calc

                # Variable-cell relaxation using UnitCellFilter
                status_text.text("Starting variable-cell relaxation...")
                ucf = UnitCellFilter(atoms)
                
                # Initialize optimizer
                opt = BFGS(ucf, logfile=None, trajectory=None)
                
                # Run relaxation with progress updates
                for i in range(max_steps):
                    status_text.text(f"Optimization step {i+1}/{max_steps}")
                    progress_bar.progress((i + 1) / max_steps)
                    
                    # Take one optimization step
                    opt.run(fmax=forc_conv_thr, steps=1)
                    
                    # Break if convergence is reached
                    if opt.converged():
                        break

                # Get final results
                status_text.text("Finalizing calculation...")
                final_energy = atoms.get_potential_energy()
                final_cell = atoms.get_cell()
                
                # Convert back to Bohr for consistency
                lattice_constants_bohr = [length / 0.529177 for length in final_cell.lengths()]
                final_volume = atoms.get_volume() / (0.529177 ** 3)  # Volume in Bohr³

                # Display results
                st.success("✅ Relaxation complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Energy", f"{final_energy:.6f} eV")
                    st.metric("Final Volume", f"{final_volume:.4f} Bohr³")
                
                with col2:
                    st.metric("Lattice constant a", f"{lattice_constants_bohr[0]:.6f} Bohr")
                    st.metric("Lattice constant b", f"{lattice_constants_bohr[1]:.6f} Bohr")
                
                with col3:
                    st.metric("Lattice constant c", f"{lattice_constants_bohr[2]:.6f} Bohr")
                    st.metric("Initial lattice", f"{alat:.4f} Bohr")

                # Save to database
                save_result(
                    element=element,
                    alat_bohr=alat,
                    ecut=ecut,
                    kpts=kpts,
                    etot_conv_thr=etot_conv_thr,
                    forc_conv_thr=forc_conv_thr,
                    energy=final_energy,
                    a=lattice_constants_bohr[0],
                    b=lattice_constants_bohr[1],
                    c=lattice_constants_bohr[2]
                )
                
                st.info(f"📊 Results saved to database")

            except Exception as e:
                st.error(f"❌ Calculation failed: {str(e)}")
                st.info("Try adjusting parameters or check system resources")

# ---------------------------------------------------------------
# Main App Layout with Tabs
# ---------------------------------------------------------------
tab_li, tab_sn, tab_db, tab_info = st.tabs([
    "🔹 Lithium (Li)", 
    "🔸 Tin (Sn)", 
    "📊 Results Database",
    "ℹ️ Instructions"
])

with tab_li:
    st.info("BCC Lithium - Recommended: alat ≈ 6.0 Bohr, Ecut ≈ 400 eV, kpts ≈ 8")
    run_vc_relax(element='Li', default_alat_bohr=6.0, default_ecut=400, default_kpts=8)

with tab_sn:
    st.info("Diamond structure Tin - Recommended: alat ≈ 8.0 Bohr, Ecut ≈ 500 eV, kpts ≈ 6")
    run_vc_relax(element='Sn', default_alat_bohr=8.0, default_ecut=500, default_kpts=6)

with tab_db:
    st.subheader("Stored Relaxation Results")
    
    # Add option to clear database
    if st.button("Clear All Results"):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM relax_results")
        conn.commit()
        conn.close()
        st.success("Database cleared!")
        st.experimental_rerun()
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT 
            id, element, 
            alat_bohr as "Initial a₀ (Bohr)",
            ecut as "Ecut (eV)", 
            kpts as "K-points",
            energy as "Energy (eV)",
            a as "a (Bohr)", 
            b as "b (Bohr)", 
            c as "c (Bohr)",
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
        summary = df.groupby('element').agg({
            'Energy (eV)': ['min', 'mean', 'max'],
            'a (Bohr)': ['mean', 'std'],
            'Volume (Bohr³)': ['mean', 'std']
        }).round(6)
        st.dataframe(summary)
    else:
        st.info("No results found. Run a calculation to populate the database.")

with tab_info:
    st.subheader("Instructions & Parameters")
    
    st.markdown("""
    ### 🎯 Recommended Parameters
    
    **Lithium (Li) - BCC structure:**
    - Initial lattice: 6.0-6.5 Bohr
    - Cutoff energy: 350-450 eV
    - K-points: 8×8×8
    
    **Tin (Sn) - Diamond structure:**
    - Initial lattice: 7.5-8.5 Bohr  
    - Cutoff energy: 450-550 eV
    - K-points: 6×6×6
    
    ### ⚙️ Parameter Guidelines
    
    - **Energy cutoff**: Higher = more accurate but slower (400-600 eV typical)
    - **K-points**: Denser = better Brillouin zone sampling
    - **Convergence**: Tighter thresholds = more accurate but slower convergence
    
    ### ⏰ Performance Notes
    
    - Calculations run entirely in Python via GPAW
    - Single-point calculations: ~1-5 minutes
    - Full relaxations: ~5-30 minutes depending on parameters
    - Larger systems require more memory and time
    """)
    
    st.subheader("Technical Details")
    st.markdown("""
    - **Method**: DFT with PBE functional
    - **Calculator**: GPAW with Plane-Wave basis
    - **Relaxation**: BFGS optimizer with UnitCellFilter for variable-cell relaxation
    - **Units**: Input/Output in Bohr (atomic units), internal calculations in Å
    """)

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown("---")
st.caption("ASE + GPAW Variable-Cell Relaxation App | Made with Streamlit")
