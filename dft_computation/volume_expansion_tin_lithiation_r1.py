import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from ase import Atoms
from ase.optimize import BFGS, LBFGS
from ase.spacegroup import crystal
from ase.filters import FrechetCellFilter
from gpaw import GPAW, PW
import os

# ============================================================================
#  UI Setup
# ============================================================================
st.set_page_config(page_title="ASE+GPAW: Fast vc-relax for Li-Sn", layout="wide")
st.title("⚡ ASE + GPAW: Fast vc-relax for Li-Sn System")

st.markdown("""
This app performs **variable-cell relaxation (vc-relax)** for Lithium-Tin systems using ASE + GPAW with  
**`FrechetCellFilter`** (the recommended modern filter for cell optimization) and interactive visualizations.

**Speed vs Accuracy Trade-offs:**  
- Lower cutoff energy → Faster but less accurate  
- Fewer k-points → Much faster but less accurate  
- Looser convergence → Faster convergence but less precise
""")

# Speed optimization controls
st.sidebar.header("⚡ Speed Optimization")
calculation_mode = st.sidebar.selectbox(
    "Calculation Mode",
    options=["Fast Testing", "Balanced", "High Accuracy"],
    index=0,
    help="Choose between speed and accuracy"
)

# Set parameters based on calculation mode
if calculation_mode == "Fast Testing":
    default_energy_conv = 1e-4
    default_force_conv = 5e-3
    ecut_factor = 0.7
    kpts_factor = 0.5
    max_steps = 50
elif calculation_mode == "Balanced":
    default_energy_conv = 1e-5
    default_force_conv = 1e-3
    ecut_factor = 0.85
    kpts_factor = 0.75
    max_steps = 100
else:  # High Accuracy
    default_energy_conv = 1e-6
    default_force_conv = 1e-4
    ecut_factor = 1.0
    kpts_factor = 1.0
    max_steps = 200

st.sidebar.write(f"**Current Mode:** {calculation_mode}")
st.sidebar.write(f"**Max Steps:** {max_steps}")

# ============================================================================
#  Structure Selection & Defaults
# ============================================================================
structure = st.selectbox("Select Structure", options=['Li (BCC)', 'Sn (diamond cubic)', 'Sn (BCT)', 'Li2Sn5'])

# Default parameters based on structure (high accuracy baseline)
if structure == 'Li (BCC)':
    default_ka_kb = max(2, int(10 * kpts_factor))
    default_kc = max(2, int(10 * kpts_factor))
    default_ecut = max(200, int(400 * ecut_factor))
    default_a = 3.49
elif structure == 'Sn (diamond cubic)':
    default_ka_kb = max(2, int(8 * kpts_factor))
    default_kc = max(2, int(8 * kpts_factor))
    default_ecut = max(200, int(500 * ecut_factor))
    default_a = 6.49
elif structure == 'Sn (BCT)':
    default_ka_kb = max(2, int(8 * kpts_factor))
    default_kc = max(2, int(12 * kpts_factor))
    default_ecut = max(200, int(500 * ecut_factor))
    default_a = 5.83
    default_c = 3.18
elif structure == 'Li2Sn5':
    default_ka_kb = max(2, int(6 * kpts_factor))
    default_kc = max(2, int(16 * kpts_factor))
    default_ecut = max(200, int(500 * ecut_factor))
    default_a = 10.274
    default_c = 3.125

# Initial parameters with speed-optimized defaults
if structure == 'Li (BCC)':
    a = st.number_input("Initial a (Å)", min_value=2.0, max_value=10.0, value=default_a)
    # Use conventional 2-atom BCC cell to avoid FrechetCellFilter issues
    atoms = Atoms('Li2',
                  positions=[[0,0,0], [0.5,0.5,0.5]],
                  cell=[a, a, a],
                  pbc=True)
    is_cubic = True
elif structure == 'Sn (diamond cubic)':
    a = st.number_input("Initial a (Å)", min_value=4.0, max_value=10.0, value=default_a)
    atoms = crystal('Sn', basis=[(0,0,0), (0.25,0.25,0.25)], spacegroup=227, cellpar=[a, a, a, 90, 90, 90])
    is_cubic = True
elif structure == 'Sn (BCT)':
    a = st.number_input("Initial a (Å)", min_value=2.0, max_value=10.0, value=default_a)
    c = st.number_input("Initial c (Å)", min_value=2.0, max_value=10.0, value=default_c)
    atoms = crystal('Sn', basis=[(0,0,0)], spacegroup=141, cellpar=[a, a, c, 90, 90, 90])
    is_cubic = False
elif structure == 'Li2Sn5':
    a = st.number_input("Initial a (Å)", min_value=5.0, max_value=15.0, value=default_a)
    c = st.number_input("Initial c (Å)", min_value=2.0, max_value=5.0, value=default_c)
    atoms = crystal(
        symbols=['Sn', 'Li', 'Sn'],
        basis=[(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)],
        spacegroup=127,
        cellpar=[a, a, c, 90, 90, 90]
    )
    is_cubic = False

# Convergence thresholds with speed-optimized defaults
col1, col2 = st.columns(2)

with col1:
    etot_conv_thr = st.number_input(
        "SCF energy convergence threshold (eV/atom)",
        min_value=1e-8,
        max_value=1e-2,
        value=default_energy_conv,
        format="%.1e",
        help="Larger values = faster but less accurate"
    )
    ecut = st.number_input(
        "Plane-wave cutoff energy (eV)",
        min_value=200,
        max_value=1000,
        value=default_ecut,
        help="Lower values = faster but less accurate"
    )

with col2:
    forc_conv_thr = st.number_input(
        "Force convergence threshold (eV/Å)",
        min_value=1e-8,
        max_value=1e-1,
        value=default_force_conv,
        format="%.1e",
        help="Larger values = faster convergence"
    )
    ka_kb = st.number_input(
        "K-points along a/b (ka x kb)",
        min_value=2,
        max_value=20,
        value=default_ka_kb,
        help="Fewer k-points = much faster"
    )
    kc = st.number_input(
        "K-points along c (kc)",
        min_value=2,
        max_value=20,
        value=default_kc
    )

# Additional speed optimizations
st.sidebar.subheader("Advanced Speed Options")
use_fast_optimizer = st.sidebar.checkbox("Use Fast Optimizer (LBFGS)", value=True)
reduce_mixer = st.sidebar.checkbox("Reduce Mixer Settings", value=True)
simple_occupations = st.sidebar.checkbox("Simple Occupations", value=True)

# Display current speed settings
st.sidebar.markdown("---")
st.sidebar.subheader("Current Speed Settings")
st.sidebar.write(f"ECut: {ecut} eV")
st.sidebar.write(f"K-points: ({ka_kb}, {ka_kb}, {kc})")
st.sidebar.write(f"Force convergence: {forc_conv_thr:.1e} eV/Å")
st.sidebar.write(f"Energy convergence: {etot_conv_thr:.1e} eV/atom")

# ============================================================================
#  Core relaxation function (using FrechetCellFilter with trajectory)
# ============================================================================
def create_fast_calculator(atoms, structure_name, ecut, ka_kb, kc):
    """Create GPAW calculator with speed/accuracy trade-offs."""
    convergence_settings = {'energy': etot_conv_thr}

    if reduce_mixer:
        from gpaw import Mixer
        mixer = Mixer(0.1, 5, 10)          # faster mixing
    else:
        mixer = None

    if simple_occupations:
        occupations = None                  # default (fast)
    else:
        from gpaw import FermiDirac
        occupations = FermiDirac(0.1)

    return GPAW(
        mode=PW(ecut),
        xc='PBE',
        kpts=(ka_kb, ka_kb, kc),
        convergence=convergence_settings,
        txt=f'{structure_name.replace(" ", "_")}_gpaw.log',
        maxiter=100,
        mixer=mixer,
        occupations=occupations,
    )

def perform_vc_relax(atoms, ecut, kpts, forc_conv_thr, max_steps=100,
                     optimizer='LBFGS', structure_name="unknown"):
    """
    Perform variable-cell relaxation using FrechetCellFilter.
    Returns a dictionary with results and trajectory data.
    """
    try:
        # Attach calculator
        calc = create_fast_calculator(atoms, structure_name, ecut, kpts[0], kpts[2])
        atoms.calc = calc

        st.write(f"Starting vc-relax for {structure_name} with FrechetCellFilter...")

        # Use FrechetCellFilter for full cell relaxation
        fcf = FrechetCellFilter(atoms, scalar_pressure=0.0)  # target zero pressure

        # Choose optimizer
        if optimizer.upper() == 'LBFGS':
            opt = LBFGS(fcf, logfile=f'{structure_name}_relax.log')
        else:
            opt = BFGS(fcf, logfile=f'{structure_name}_relax.log')

        # Store trajectory for plotting
        energy_steps = []
        force_steps = []

        def update_trajectory():
            """Called after each optimization step."""
            try:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                max_force = np.max(np.abs(forces))
                energy_steps.append(energy)
                force_steps.append(max_force)
            except Exception:
                pass

        opt.attach(update_trajectory, interval=1)

        # Run relaxation
        opt.run(fmax=forc_conv_thr, steps=max_steps)

        # Gather final results
        final_cell = atoms.get_cell()
        lattice = final_cell.lengths()
        volume = atoms.get_volume()
        total_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        max_force = np.max(np.abs(final_forces))
        stress = atoms.get_stress()          # Voigt order: xx, yy, zz, yz, xz, xy
        hydrostatic_pressure = -np.trace(stress[:3]) / 3   # eV/Å³

        # Count Sn atoms (if any)
        n_sn = sum(1 for atom in atoms if atom.symbol == 'Sn')

        return {
            'atoms': atoms,
            'lattice': lattice,
            'volume': volume,
            'energy': total_energy,
            'max_force': max_force,
            'stress': stress,
            'hydrostatic_pressure': hydrostatic_pressure,
            'n_sn': n_sn,
            'volume_per_sn': volume / n_sn if n_sn > 0 else None,
            'energy_steps': energy_steps,
            'force_steps': force_steps,
        }
    except Exception as e:
        st.error(f"vc-relax failed for {structure_name}: {e}")
        return None

# ============================================================================
#  Main vc-relax calculation
# ============================================================================
run_calc = st.button("🚀 Run Fast vc-relax Calculation")

if run_calc:
    try:
        st.write(f"Setting up **fast** vc-relax for {structure}...")
        st.write(f"**Mode:** {calculation_mode}")

        # Show parameters
        with st.expander("Current Optimization Parameters"):
            st.write(f"- Plane-wave cutoff: {ecut} eV")
            st.write(f"- K-points grid: ({ka_kb}, {ka_kb}, {kc})")
            st.write(f"- Force convergence: {forc_conv_thr:.1e} eV/Å")
            st.write(f"- Energy convergence: {etot_conv_thr:.1e} eV/atom")
            st.write(f"- Maximum steps: {max_steps}")

        # Run relaxation
        kpts_tuple = (ka_kb, ka_kb, kc)
        result = perform_vc_relax(
            atoms=atoms.copy(),   # avoid modifying original
            ecut=ecut,
            kpts=kpts_tuple,
            forc_conv_thr=forc_conv_thr,
            max_steps=max_steps,
            optimizer='LBFGS' if use_fast_optimizer else 'BFGS',
            structure_name=structure
        )

        if result:
            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Structural Results")
                st.write(f"**Optimized lattice constants (Å):** {result['lattice']}")
                st.write(f"**Volume (Å³):** {result['volume']:.6f}")
                if result['volume_per_sn']:
                    st.write(f"**Volume per Sn (Å³):** {result['volume_per_sn']:.6f}")
                st.write(f"**Final max force:** {result['max_force']:.6f} eV/Å")
                st.write(f"**Hydrostatic stress (GPa):** {result['hydrostatic_pressure'] * 160.21766208:.4f} GPa")

            with col2:
                st.subheader("Energetic Results")
                st.write(f"**Total energy (eV):** {result['energy']:.6f}")
                st.write(f"**Energy/atom (eV):** {result['energy']/len(atoms):.6f}")
                st.write(f"**Convergence:** {'Yes' if result['max_force'] <= forc_conv_thr else 'No'}")

            if result['max_force'] > forc_conv_thr:
                st.warning("Calculation stopped before full convergence. Consider increasing max steps or loosening convergence criteria.")

            # --- Visualizations ---
            st.subheader("📊 Visualization")

            # 1. Energy vs Step line plot
            if result['energy_steps']:
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Scatter(
                    x=list(range(1, len(result['energy_steps'])+1)),
                    y=result['energy_steps'],
                    mode='lines+markers',
                    name='Total Energy'
                ))
                fig_energy.update_layout(
                    title="Energy vs Optimization Step",
                    xaxis_title="Step Number",
                    yaxis_title="Total Energy (eV)"
                )
                st.plotly_chart(fig_energy, use_container_width=True)

            # 2. Force vs Step line plot
            if result['force_steps']:
                fig_force = go.Figure()
                fig_force.add_trace(go.Scatter(
                    x=list(range(1, len(result['force_steps'])+1)),
                    y=result['force_steps'],
                    mode='lines+markers',
                    name='Max Force'
                ))
                fig_force.update_layout(
                    title="Max Force vs Optimization Step",
                    xaxis_title="Step Number",
                    yaxis_title="Max Force (eV/Å)"
                )
                st.plotly_chart(fig_force, use_container_width=True)

            # 3. Stress tensor bar chart
            stress_components = ['σ_xx', 'σ_yy', 'σ_zz', 'σ_yz', 'σ_xz', 'σ_xy']
            stress_values = result['stress']  # in eV/Å³
            stress_gpa = stress_values * 160.21766208  # convert to GPa

            fig_stress = go.Figure()
            fig_stress.add_trace(go.Bar(
                x=stress_components,
                y=stress_gpa,
                text=[f"{v:.2f}" for v in stress_gpa],
                textposition='auto'
            ))
            fig_stress.update_layout(
                title="Stress Tensor Components (GPa)",
                yaxis_title="Stress (GPa)"
            )
            st.plotly_chart(fig_stress, use_container_width=True)

            # 4. Radar chart: convergence quality
            categories = ['ECut (norm)', 'K-point density (norm)', 'Force conv.', 'Energy conv.', '1/Max force', '1/Residual stress']
            # Normalize values to [0,1] for radar (higher is better)
            ecut_norm = min(1.0, ecut / 1000.0)                     # 1000 eV → 1
            kpt_norm = min(1.0, (ka_kb * kc) / 400.0)              # 20x20x20=8000, we cap at 400
            force_conv_norm = min(1.0, 1.0 / (forc_conv_thr * 100)) # 1e-4 → 100, capped at 1
            energy_conv_norm = min(1.0, 1.0 / (etot_conv_thr * 1e4)) # 1e-6 → 100
            max_force_norm = min(1.0, 0.1 / (result['max_force'] + 1e-8))
            stress_norm = min(1.0, 0.1 / (abs(result['hydrostatic_pressure']) + 1e-8))

            radar_values = [ecut_norm, kpt_norm, force_conv_norm,
                            energy_conv_norm, max_force_norm, stress_norm]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_values,
                theta=categories,
                fill='toself',
                name='Current Run'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Convergence Quality (higher = better)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    except Exception as e:
        st.error(f"Calculation failed: {e}")

# ============================================================================
#  Volume Expansion Analysis (with consistent FrechetCellFilter)
# ============================================================================
st.header("⚡ Fast Volume Expansion Analysis")

def run_fast_expansion_calculation():
    """Run volume expansion with FrechetCellFilter for both phases."""
    try:
        # Get settings based on mode
        if calculation_mode == "Fast Testing":
            ecut_exp = 350
            kpts_sn = (4, 4, 6)
            kpts_li2sn5 = (3, 3, 8)
            forc_conv_exp = 5e-3
            max_steps_exp = 30
        elif calculation_mode == "Balanced":
            ecut_exp = 450
            kpts_sn = (6, 6, 8)
            kpts_li2sn5 = (4, 4, 12)
            forc_conv_exp = 1e-3
            max_steps_exp = 60
        else:
            ecut_exp = 500
            kpts_sn = (8, 8, 12)
            kpts_li2sn5 = (6, 6, 16)
            forc_conv_exp = 1e-4
            max_steps_exp = 100

        st.write("### Phase 1: BCT Sn vc-relax (FrechetCellFilter)")
        atoms_sn = crystal('Sn', basis=[(0,0,0)], spacegroup=141,
                           cellpar=[5.83, 5.83, 3.18, 90, 90, 90])

        result_sn = perform_vc_relax(
            atoms=atoms_sn,
            ecut=ecut_exp,
            kpts=kpts_sn,
            forc_conv_thr=forc_conv_exp,
            max_steps=max_steps_exp,
            structure_name="BCT_Sn"
        )

        st.write("### Phase 2: Li₂Sn₅ vc-relax (FrechetCellFilter)")
        atoms_li2sn5 = crystal(
            symbols=['Sn', 'Li', 'Sn'],
            basis=[(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)],
            spacegroup=127,
            cellpar=[10.274, 10.274, 3.125, 90, 90, 90]
        )

        result_li2sn5 = perform_vc_relax(
            atoms=atoms_li2sn5,
            ecut=ecut_exp,
            kpts=kpts_li2sn5,
            forc_conv_thr=forc_conv_exp,
            max_steps=max_steps_exp,
            structure_name="Li2Sn5"
        )

        if result_sn and result_li2sn5:
            v_sn = result_sn['volume_per_sn']
            v_li2sn5 = result_li2sn5['volume_per_sn']
            expansion = (v_li2sn5 - v_sn) / v_sn * 100

            st.success("### Volume Expansion Results (per Sn atom)")

            # Bar chart
            df_exp = pd.DataFrame({
                'Phase': ['BCT Sn', 'Li₂Sn₅'],
                'Volume per Sn (Å³)': [v_sn, v_li2sn5]
            })
            fig_bar = px.bar(df_exp, x='Phase', y='Volume per Sn (Å³)',
                             text=df_exp['Volume per Sn (Å³)'].round(3),
                             title="Volume per Sn Atom")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("BCT Sn Volume/Sn", f"{v_sn:.4f} Å³")
                st.metric("Li₂Sn₅ Volume/Sn", f"{v_li2sn5:.4f} Å³")
            with col2:
                st.metric("Volume Expansion", f"{expansion:.2f}%", delta=f"{expansion:.2f}%")

            st.info(f"**Calculation Mode:** {calculation_mode}  |  **Filter:** FrechetCellFilter")
            st.info("For production runs, use 'High Accuracy' mode.")
        else:
            st.error("One or both relaxations failed. Check logs for details.")

    except Exception as e:
        st.error(f"Fast expansion calculation failed: {e}")

run_fast_expansion_button = st.button("🚀 Compute Fast Volume Expansion")

if run_fast_expansion_button:
    st.info(f"Starting fast volume expansion in {calculation_mode} mode...")
    run_fast_expansion_calculation()

# ============================================================================
#  Tips and Performance Estimates
# ============================================================================
with st.expander("💡 Additional Speed Optimization Tips"):
    st.markdown("""
    **For Maximum Speed (Testing Only):**
    - Use **ECut = 300 eV** and **k-points = (2,2,2)**
    - Set **force convergence = 0.01 eV/Å**
    - Set **energy convergence = 1e-3 eV/atom**
    - Use **LBFGS optimizer** with **max steps = 20**

    **Expected Speedup:** 5-10x faster than high-accuracy settings

    **Warning:** Results will be qualitative only – use for initial testing and prototyping
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Performance Estimates")
if calculation_mode == "Fast Testing":
    st.sidebar.write("**Estimate:** 2-10 minutes")
    st.sidebar.write("**Use case:** Quick testing")
elif calculation_mode == "Balanced":
    st.sidebar.write("**Estimate:** 10-60 minutes")
    st.sidebar.write("**Use case:** Development")
else:
    st.sidebar.write("**Estimate:** 1-6 hours")
    st.sidebar.write("**Use case:** Production")
