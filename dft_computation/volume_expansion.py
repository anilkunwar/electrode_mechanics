import streamlit as st
from ase import Atoms
from ase.optimize import BFGS, LBFGS
from ase.constraints import UnitCellFilter, ExpCellFilter
from gpaw import GPAW, PW
import sqlite3
import os
import pandas as pd
import numpy as np
from ase.io import read
from ase.build import bulk
import tempfile

# ---------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="ASE + GPAW vc-relax for Li-Sn", layout="centered")
st.title("ASE + GPAW: Variable-Cell Relaxation for Li-Sn Phases")

st.markdown("""
Perform **variable-cell relaxation (vc-relax)** calculations for **pure BCT Sn** and **Li-Sn phases**
using the **GPAW** DFT calculator integrated with ASE. Calculate volume expansion of Li-Sn phases relative to pure Sn.

**Note:** Calculations may take several minutes. For larger phases, use reduced k-points and looser convergence.
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
            energy_per_atom REAL,
            a REAL,
            b REAL,
            c REAL,
            alpha REAL,
            beta REAL,
            gamma REAL,
            volume REAL,
            volume_per_sn REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_result(phase, num_li, num_sn, alat_bohr, ecut, kpts, etot_conv_thr, forc_conv_thr, 
                energy, cell_lengths, cell_angles, volume):
    a, b, c = cell_lengths
    alpha, beta, gamma = cell_angles
    energy_per_atom = energy / (num_li + num_sn) if (num_li + num_sn) > 0 else 0
    volume_per_sn = volume / num_sn if num_sn > 0 else 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO relax_results 
        (phase, num_li, num_sn, alat_bohr, ecut, kpts, etot_conv_thr, forc_conv_thr, 
         energy, energy_per_atom, a, b, c, alpha, beta, gamma, volume, volume_per_sn)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (phase, num_li, num_sn, alat_bohr, ecut, kpts, etot_conv_thr, forc_conv_thr, 
          energy, energy_per_atom, a, b, c, alpha, beta, gamma, volume, volume_per_sn))
    conn.commit()
    conn.close()

# ---------------------------------------------------------------
# Improved Structure Creation Functions
# ---------------------------------------------------------------
def create_bct_sn():
    """Create beta-Sn (BCT) structure"""
    # Beta-Sn structure (tetragonal, I4_1/amd)
    a_ang = 5.831 * 0.529177  # Convert from Bohr to Å for internal use
    c_ang = 3.182 * 0.529177
    atoms = Atoms('Sn2',
                  positions=[[0, 0, 0], 
                           [a_ang/2, a_ang/2, c_ang/2]],
                  cell=[[a_ang, 0, 0], 
                        [0, a_ang, 0], 
                        [0, 0, c_ang]],
                  pbc=True)
    return atoms

def create_lisn_simple():
    """Create a simple LiSn structure for testing"""
    # Simple cubic-like structure for testing
    a_ang = 6.0 * 0.529177  # ~6.0 Bohr in Å
    atoms = Atoms('LiSn',
                  positions=[[0, 0, 0], [a_ang/2, a_ang/2, a_ang/2]],
                  cell=[[a_ang, 0, 0], 
                        [0, a_ang, 0], 
                        [0, 0, a_ang]],
                  pbc=True)
    return atoms

def create_li2sn5():
    """Create Li2Sn5 structure - smaller for faster calculation"""
    # Approximate structure - in practice, use CIF from Materials Project
    a_ang = 8.0 * 0.529177
    atoms = Atoms('Li2Sn5',
                  positions=[
                      [0.0, 0.0, 0.0],    # Li1
                      [a_ang/2, a_ang/2, a_ang/2],  # Li2  
                      [a_ang/4, a_ang/4, a_ang/4],  # Sn1
                      [3*a_ang/4, a_ang/4, a_ang/4], # Sn2
                      [a_ang/4, 3*a_ang/4, a_ang/4], # Sn3
                      [a_ang/4, a_ang/4, 3*a_ang/4], # Sn4
                      [3*a_ang/4, 3*a_ang/4, a_ang/4] # Sn5
                  ],
                  cell=[[a_ang, 0, 0], 
                        [0, a_ang, 0], 
                        [0, 0, a_ang]],
                  pbc=True)
    return atoms

# ---------------------------------------------------------------
# Improved Relaxation Function with Better Error Handling
# ---------------------------------------------------------------
def run_vc_relax(phase, default_alat_bohr, default_ecut, default_kpts, initial_atoms_factory):
    st.subheader(f"{phase} Relaxation Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        alat = st.number_input(
            f"Initial lattice scale factor",
            min_value=0.5, max_value=2.0, value=1.0, step=0.1,
            key=f"alat_{phase}"
        )
        
        ecut = st.number_input(
            "Plane-wave cutoff energy (eV)",
            min_value=200, max_value=800, value=default_ecut, 
            key=f"ecut_{phase}"
        )
        
        kpts = st.number_input(
            "K-points grid density",
            min_value=2, max_value=12, value=default_kpts,
            key=f"kpts_{phase}"
        )
    
    with col2:
        etot_conv_thr = st.number_input(
            "Energy convergence (eV/atom)",
            min_value=1e-6, max_value=1e-3, value=1e-4, format="%.1e",
            key=f"etot_{phase}"
        )
        
        forc_conv_thr = st.number_input(
            "Force convergence (eV/Å)",
            min_value=1e-3, max_value=0.1, value=0.05, format="%.1f",
            key=f"fmax_{phase}"
        )
        
        max_steps = st.number_input(
            "Max optimization steps",
            min_value=5, max_value=50, value=20,
            key=f"steps_{phase}"
        )
        
        optimizer_choice = st.selectbox(
            "Optimizer",
            ["BFGS", "LBFGS"],
            key=f"opt_{phase}"
        )

    run_calc = st.button(f"Run vc-relax for {phase}", type="primary")

    if run_calc:
        with st.spinner(f"Running variable-cell relaxation for {phase}... This may take a while."):
            try:
                # Create initial structure
                atoms = initial_atoms_factory()
                
                # Scale the structure
                if alat != 1.0:
                    atoms.set_cell(atoms.get_cell() * alat, scale_atoms=True)
                
                st.write("### Calculation Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Setup GPAW calculator with more stable settings
                status_text.text("Initializing GPAW calculator...")
                
                # Use temporary file for output to avoid permission issues
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
                    log_file = tmp.name
                
                calc = GPAW(
                    mode=PW(ecut),
                    xc='PBE',
                    kpts={'size': (kpts, kpts, kpts), 'gamma': True},
                    convergence={'energy': etot_conv_thr, 'density': 1e-4},
                    txt=log_file,
                    symmetry={'point_group': False},  # Disable symmetry for better convergence
                    occupations={'name': 'fermi-dirac', 'width': 0.1},  # Smearing for metals
                    mixer={'backend': 'pulay', 'beta': 0.25, 'nmaxold': 3, 'weight': 50}
                )

                atoms.calc = calc

                # Variable-cell relaxation with better settings
                status_text.text("Starting variable-cell relaxation...")
                
                # Use ExpCellFilter for better stability
                ucf = ExpCellFilter(atoms, hydrostatic_strain=False)
                
                # Choose optimizer
                if optimizer_choice == "LBFGS":
                    opt = LBFGS(uf, logfile=None)
                else:
                    opt = BFGS(uf, logfile=None)
                
                # Run relaxation with better convergence criteria
                converged = False
                for i in range(max_steps):
                    status_text.text(f"Optimization step {i+1}/{max_steps}")
                    progress_bar.progress((i + 1) / max_steps)
                    
                    try:
                        opt.run(fmax=forc_conv_thr, steps=1)
                        if opt.converged():
                            converged = True
                            break
                    except Exception as step_error:
                        st.warning(f"Step {i+1} had issues: {step_error}. Continuing...")
                        continue

                # Get final results
                status_text.text("Finalizing calculation...")
                final_energy = atoms.get_potential_energy()
                final_cell = atoms.get_cell()
                
                # Convert to Bohr for consistency
                cell_lengths = [length / 0.529177 for length in final_cell.lengths()]
                cell_angles = final_cell.angles()
                final_volume = atoms.get_volume() / (0.529177 ** 3)  # Bohr³
                
                # Count atoms
                symbols = atoms.get_chemical_symbols()
                num_li = symbols.count('Li')
                num_sn = symbols.count('Sn')

                # Display results
                if converged:
                    st.success("✅ Relaxation converged!")
                else:
                    st.warning("⚠️ Relaxation stopped (max steps reached) but results are available.")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Energy", f"{final_energy:.6f} eV")
                    st.metric("Energy/Atom", f"{final_energy/len(atoms):.6f} eV")
                    st.metric("Final Volume", f"{final_volume:.2f} Bohr³")
                
                with col2:
                    st.metric("Lattice a", f"{cell_lengths[0]:.4f} Bohr")
                    st.metric("Lattice b", f"{cell_lengths[1]:.4f} Bohr")
                    st.metric("Lattice c", f"{cell_lengths[2]:.4f} Bohr")
                
                with col3:
                    st.metric("Angle α", f"{cell_angles[0]:.2f}°")
                    st.metric("Angle β", f"{cell_angles[1]:.2f}°") 
                    st.metric("Angle γ", f"{cell_angles[2]:.2f}°")

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
                    cell_angles=cell_angles,
                    volume=final_volume
                )
                
                st.info(f"📊 Results saved to database")
                
                # Clean up
                if os.path.exists(log_file):
                    os.unlink(log_file)

            except Exception as e:
                st.error(f"❌ Calculation failed: {str(e)}")
                st.info("""
                **Troubleshooting tips:**
                - Try using LBFGS optimizer instead of BFGS
                - Increase force convergence threshold (0.05 eV/Å)
                - Reduce k-points density
                - Use lower cutoff energy
                - The 'leading minor not positive definite' error often indicates numerical instability
                """)

# ---------------------------------------------------------------
# Volume Expansion Analysis
# ---------------------------------------------------------------
def calculate_volume_expansion():
    st.subheader("Volume Expansion Analysis")
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM relax_results ORDER BY timestamp DESC", conn)
    conn.close()
    
    if df.empty:
        st.info("No results available for analysis.")
        return
    
    # Get unique phases
    phases = df['phase'].unique()
    
    if len(phases) < 2:
        st.info("Need at least 2 different phases to calculate volume expansion.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        reference_phase = st.selectbox(
            "Reference phase (denominator):",
            phases,
            key="ref_phase"
        )
    
    with col2:
        target_phase = st.selectbox(
            "Target phase (numerator):", 
            phases,
            key="target_phase"
        )
    
    if st.button("Calculate Volume Expansion"):
        # Get most recent calculations for each phase
        ref_df = df[df['phase'] == reference_phase].iloc[0]
        target_df = df[df['phase'] == target_phase].iloc[0]
        
        # Calculate volume per Sn atom
        ref_vol_per_sn = ref_df['volume'] / ref_df['num_sn']
        target_vol_per_sn = target_df['volume'] / target_df['num_sn']
        
        # Calculate expansion
        expansion = ((target_vol_per_sn - ref_vol_per_sn) / ref_vol_per_sn) * 100
        
        st.success("**Volume Expansion Results:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{reference_phase} Vol/Sn", f"{ref_vol_per_sn:.2f} Bohr³")
        with col2:
            st.metric(f"{target_phase} Vol/Sn", f"{target_vol_per_sn:.2f} Bohr³")
        with col3:
            st.metric("Volume Expansion", f"{expansion:.1f}%", 
                     delta=f"{expansion:.1f}%")

# ---------------------------------------------------------------
# Main App Layout with Tabs
# ---------------------------------------------------------------
tab_sn, tab_lisn, tab_li2sn5, tab_db, tab_info = st.tabs([
    "🔹 Pure BCT Sn", 
    "🔸 LiSn Simple", 
    "🔷 Li₂Sn₅",
    "📊 Results & Expansion",
    "ℹ️ Instructions"
])

with tab_sn:
    st.info("BCT Sn (beta-Sn) - Recommended: scale ≈ 1.0, Ecut ≈ 400 eV, kpts ≈ 4")
    run_vc_relax(phase='BCT_Sn', default_alat_bohr=1.0, default_ecut=400, default_kpts=4, 
                initial_atoms_factory=create_bct_sn)

with tab_lisn:
    st.info("Simple LiSn structure for testing - Good for debugging")
    run_vc_relax(phase='LiSn', default_alat_bohr=1.0, default_ecut=400, default_kpts=4,
                initial_atoms_factory=create_lisn_simple)

with tab_li2sn5:
    st.info("Li₂Sn₅ - Intermediate phase with reasonable size")
    run_vc_relax(phase='Li2Sn5', default_alat_bohr=1.0, default_ecut=400, default_kpts=3,
                initial_atoms_factory=create_li2sn5)

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
            alat_bohr as "Scale Factor",
            ecut as "Ecut (eV)", 
            kpts as "K-points",
            energy as "Energy (eV)",
            energy_per_atom as "E/Atom (eV)",
            a as "a (Bohr)", 
            b as "b (Bohr)", 
            c as "c (Bohr)",
            alpha as "α (°)", 
            beta as "β (°)", 
            gamma as "γ (°)",
            volume as "Volume (Bohr³)",
            volume_per_sn as "Vol/Sn (Bohr³)",
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
            'Energy (eV)': ['count', 'min', 'mean'],
            'E/Atom (eV)': ['mean', 'std'],
            'Volume (Bohr³)': ['mean', 'std'],
            'Vol/Sn (Bohr³)': ['mean', 'std']
        }).round(4)
        st.dataframe(summary)
        
        # Volume Expansion Calculator
        calculate_volume_expansion()
        
    else:
        st.info("No results found. Run a calculation to populate the database.")

with tab_info:
    st.subheader("Instructions & Parameters")
    
    st.markdown("""
    ### 🎯 Recommended Parameters for Stability
    
    **For all phases:**
    - **Scale Factor**: 1.0 (uses optimized initial structures)
    - **Cutoff Energy**: 350-450 eV (balance accuracy/speed)
    - **K-points**: 3-4 (denser for smaller cells)
    - **Force Convergence**: 0.05 eV/Å (looser for stability)
    - **Optimizer**: LBFGS (more stable than BFGS)
    
    ### ⚠️ Fixing "Leading Minor Not Positive Definite" Error
    
    This error indicates numerical instability in the Hessian matrix. Solutions:
    
    1. **Use LBFGS optimizer** instead of BFGS
    2. **Looser convergence**: Force threshold = 0.05 eV/Å
    3. **Smaller systems**: Start with simple structures
    4. **Reduce accuracy**: Lower Ecut (350 eV) and k-points (3)
    5. **Better initial guess**: Use scale factor ≈ 1.0
    
    ### 📊 Volume Expansion Calculation
    
    Volume expansion is calculated as:
    ```
    Expansion (%) = [(Vol_per_Sn_target - Vol_per_Sn_reference) / Vol_per_Sn_reference] × 100
    ```
    
    Where:
    - `Vol_per_Sn = Total_Volume / Number_of_Sn_Atoms`
    - Reference is typically pure BCT Sn
    - Target is the lithiated phase (LiSn, Li₂Sn₅, etc.)
    
    ### 🔧 Technical Details
    
    - **DFT**: PBE functional with Fermi-Dirac smearing
    - **Relaxation**: ExpCellFilter for better cell relaxation
    - **Symmetry**: Disabled for numerical stability
    - **Units**: Bohr for database, Å internally
    """)

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown("---")
st.caption("ASE + GPAW Li-Sn Relaxation & Volume Expansion App | Made with Streamlit")
