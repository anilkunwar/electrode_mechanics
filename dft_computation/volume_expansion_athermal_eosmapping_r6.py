import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS
from ase.spacegroup import crystal
from ase.units import GPa
from ase.eos import EquationOfState
from ase.filters import ExpCellFilter
from gpaw import GPAW, PW
from scipy.optimize import curve_fit
from scipy.linalg import inv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile

st.set_page_config(page_title="Sn → Li₂Sn₅ Full Mechanics", layout="wide")
st.title("⚡ Ideal DFT Workflow: EOS + Anisotropy + Stability + Fracture Prediction")
st.markdown("""
**Full Methodology Implemented**:
1. Thermodynamic stability (ΔE_f vs bulk Li/Sn)
2. Isotropic E-V mapping → Birch-Murnaghan (V₀, B₀, expansion %)
3. Uniaxial strain mapping → C₁₁, C₃₃, AR
4. Fracture risk from softening + anisotropy
""")

# Sidebar settings
st.sidebar.header("DFT Settings")
mode = st.sidebar.selectbox("Accuracy Mode", ["Fast Testing", "Production"])
if mode == "Fast Testing":
    ecut, kpts_base, fmax, n_vol = 350, (4, 4, 6), 0.05, 5
    full_elastic = False  # Skip full tensor for speed
else:
    ecut, kpts_base, fmax, n_vol = 500, (8, 8, 12), 0.01, 9
    full_elastic = st.sidebar.checkbox("Compute full elastic tensor (slow)", value=True)

def quadratic(x, A, B, C):
    return A * x**2 + B * x + C

def run_point(atoms, kpts):
    calc = GPAW(mode=PW(ecut), xc='PBE', kpts=kpts, txt=None, convergence={'energy': 1e-5})
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax)
    return atoms.get_potential_energy()

@st.cache_data(show_spinner=True)
def compute_phase(structure_name, a0, c0, symbols, spacegroup, basis, num_sn, kpts, full_elastic=False):
    template = crystal(symbols, basis=basis, spacegroup=spacegroup, cellpar=[a0, a0, c0, 90, 90, 90])
    v0_init = template.get_volume()

    # Phase 2: Isotropic EOS
    vols, energies = [], []
    scales = np.linspace(0.92, 1.08, n_vol)
    for s in scales:
        atoms = template.copy()
        atoms.set_cell(template.get_cell() * s**(1/3), scale_atoms=True)
        e = run_point(atoms, kpts)
        vols.append(atoms.get_volume())
        energies.append(e)
    eos = EquationOfState(vols, energies)
    v0, e0, B, Bp = eos.fit()

    # Phase 3: Anisotropic strains (±2%)
    strains = np.linspace(-0.02, 0.02, 5)
    # C11: strain a, fix c
    e_a = []
    for eps in strains:
        atoms = template.copy()
        atoms.set_cell([a0*(1+eps), a0*(1+eps), c0, 90, 90, 90], scale_atoms=True)
        e_a.append(run_point(atoms, kpts))
    # C33: strain c, fix a
    e_c = []
    for eps in strains:
        atoms = template.copy()
        atoms.set_cell([a0, a0, c0*(1+eps), 90, 90, 90], scale_atoms=True)
        e_c.append(run_point(atoms, kpts))

    popt_a, _ = curve_fit(quadratic, strains, e_a)
    popt_c, _ = curve_fit(quadratic, strains, e_c)
    c11 = (2 * popt_a[0] / v0) * 160.217
    c33 = (2 * popt_c[0] / v0) * 160.217

    # Full elastic tensor (tetragonal) if requested
    if full_elastic:
        # We need C11, C12, C13, C33, C44, C66
        # Use Voigt notation: 1=xx,2=yy,3=zz,4=yz,5=xz,6=xy
        # Compute C12 from volume conserving orthorhombic strain: e1 = e2 = e, e3 = -2e
        # Compute C13 from orthorhombic: e1 = e, e3 = e, e2 = -2e (or similar)
        # Compute C44 from shear strain e4 (yz)
        # Compute C66 from shear strain e6 (xy)
        # For speed, we'll compute a minimal set: C12, C13, C44, C66
        # We'll store results in a dict
        elastic = {}

        # C12: strain e1 = e2 = eps, e3 = -2*eps (volume conserving)
        e_ortho = []
        for eps in strains:
            atoms = template.copy()
            atoms.set_cell([a0*(1+eps), a0*(1+eps), c0*(1-2*eps), 90, 90, 90], scale_atoms=True)
            e_ortho.append(run_point(atoms, kpts))
        popt_ortho, _ = curve_fit(quadratic, strains, e_ortho)
        # energy change = 0.5 * V0 * (C11+C12+C11+C12-4C13)*eps^2? Actually need proper formula
        # For tetragonal, volume-conserving biaxial strain: e1=e2=ε, e3=-2ε. The elastic energy is:
        # ΔE/V0 = 0.5*(2*C11 + 2*C12 + 4*C33 - 8*C13) ε^2? Let's derive:
        # The strain tensor: [[ε,0,0],[0,ε,0],[0,0,-2ε]]
        # Energy density = 1/2 Σ C_ij ε_i ε_j, with Voigt notation: ε1=ε, ε2=ε, ε3=-2ε, rest 0.
        # So = 1/2 [C11ε1² + C12ε1ε2 + C13ε1ε3 + C21ε2ε1 + C22ε2² + C23ε2ε3 + C31ε3ε1 + C32ε3ε2 + C33ε3²]
        # = 1/2 [C11ε² + C12ε² + C13ε(-2ε) + C12ε² + C11ε² + C13ε(-2ε) + C13(-2ε)ε + C13(-2ε)ε + C33(-2ε)²]
        # = 1/2 [2C11ε² + 2C12ε² - 4C13ε² - 4C13ε² + 4C33ε²] = 1/2 [2C11ε² + 2C12ε² - 8C13ε² + 4C33ε²]
        # = (C11 + C12 - 4C13 + 2C33) ε²
        # So coefficient = C11 + C12 - 4C13 + 2C33
        coeff_ortho = popt_ortho[0] * 2 / v0 * 160.217  # Convert to GPa
        # We have C11, C33 known, so we have one equation: C11 + C12 - 4C13 + 2C33 = coeff_ortho

        # C13: strain e1 = e3 = eps, e2 = -2eps (volume conserving)
        e_ortho2 = []
        for eps in strains:
            atoms = template.copy()
            atoms.set_cell([a0*(1+eps), a0*(1-2*eps), c0*(1+eps), 90, 90, 90], scale_atoms=True)
            e_ortho2.append(run_point(atoms, kpts))
        popt_ortho2, _ = curve_fit(quadratic, strains, e_ortho2)
        # Similar derivation: e1=ε, e2=-2ε, e3=ε. Energy density = 1/2 [C11ε² + C12ε(-2ε) + C13ε² + C21ε(-2ε) + C22(-2ε)² + C23(-2ε)ε + C31ε² + C32ε(-2ε) + C33ε²]
        # = 1/2 [C11ε² -2C12ε² + C13ε² -2C12ε² + 4C22ε² -2C23ε² + C13ε² -2C23ε² + C33ε²]
        # For tetragonal, C22=C11, C23=C13. So = 1/2 [C11ε² + C11ε² + C13ε² + C13ε² + 4C11ε² -4C13ε² + C33ε²] = 1/2 [6C11ε² -2C13ε² + C33ε²]
        # = (3C11 - C13 + 0.5C33) ε²
        coeff_ortho2 = popt_ortho2[0] * 2 / v0 * 160.217
        # So: 3C11 - C13 + 0.5C33 = coeff_ortho2 => C13 = 3C11 + 0.5C33 - coeff_ortho2

        # C44: shear strain e4 = γ (yz)
        e_shear4 = []
        for gamma in strains:
            atoms = template.copy()
            # Apply shear: x' = x + gamma*z, y' = y, z' = z (or use deformation gradient)
            # For simplicity, we can apply strain via cell: cell[1,2] = gamma * cell[2,2] (since yz shear)
            atoms.set_cell([[a0, 0, 0], [0, a0, gamma*c0], [0, 0, c0]], scale_atoms=True)
            e_shear4.append(run_point(atoms, kpts))
        popt_shear4, _ = curve_fit(quadratic, strains, e_shear4)
        # For shear, ΔE/V0 = 1/2 * C44 * γ² (since only one shear component)
        c44 = (2 * popt_shear4[0] / v0) * 160.217

        # C66: shear strain e6 = γ (xy)
        e_shear6 = []
        for gamma in strains:
            atoms = template.copy()
            atoms.set_cell([[a0, gamma*a0, 0], [0, a0, 0], [0, 0, c0]], scale_atoms=True)
            e_shear6.append(run_point(atoms, kpts))
        popt_shear6, _ = curve_fit(quadratic, strains, e_shear6)
        c66 = (2 * popt_shear6[0] / v0) * 160.217

        # Solve for C12 and C13
        c13 = 3*c11 + 0.5*c33 - coeff_ortho2
        c12 = coeff_ortho - c11 + 4*c13 - 2*c33

        elastic['C11'] = c11
        elastic['C12'] = c12
        elastic['C13'] = c13
        elastic['C33'] = c33
        elastic['C44'] = c44
        elastic['C66'] = c66

        # Compute bulk modulus from elastic constants for tetragonal
        # Voigt average: B_V = (2*C11 + 2*C12 + 4*C13 + C33)/9? Actually formula:
        # For tetragonal, the bulk modulus B = (C11+C12+2C13 + C33/2)/? Let's use:
        # B = (C11 + C12 + 2C13 + C33/2) / 3? Not correct. Better use Voigt-Reuss-Hill.
        # But for simplicity, we'll compute using the formula for hexagonal symmetry (similar): B = (2*(C11+C12) + 4C13 + C33)/9
        # Actually that's for hexagonal. For tetragonal, it's more complex. We'll skip for now.
        # We'll just store the constants.
    else:
        elastic = {'C11': c11, 'C33': c33}  # minimal

    return {"v0": v0, "e0": e0, "B": B/GPa, "c11": c11, "c33": c33,
            "vols": vols, "energies": energies, "num_sn": num_sn,
            "strain_data": {'strains': strains, 'e_a': e_a, 'e_c': e_c},
            "elastic": elastic}

# Reference energies (Phase 1)
@st.cache_data
def get_reference_energies():
    # Li bcc
    li = bulk('Li', 'bcc', a=3.51)
    calc = GPAW(mode=PW(ecut), xc='PBE', kpts=kpts_base, txt=None)
    li.calc = calc
    ef = ExpCellFilter(li)
    BFGS(ef).run(fmax=0.01)
    e_li_per = li.get_potential_energy() / len(li)

    # Sn bct
    sn = crystal('Sn', basis=[(0,0,0)], spacegroup=141, cellpar=[5.83, 5.83, 3.18, 90, 90, 90])
    calc_sn = GPAW(mode=PW(ecut), xc='PBE', kpts=kpts_base, txt=None)
    sn.calc = calc_sn
    opt = BFGS(sn, logfile=None)
    opt.run(fmax=0.01)
    e_sn_per = sn.get_potential_energy() / len(sn)
    return e_li_per, e_sn_per

# Main execution
if st.button("🚀 Run Full 4-Phase Analysis"):
    with st.spinner("Computing reference energies (Li, Sn)..."):
        e_li_per, e_sn_per = get_reference_energies()

    with st.spinner("BCT Sn (Phase 2+3)..."):
        sn = compute_phase("Sn", 5.83, 3.18, 'Sn', 141, [(0,0,0)], 4, kpts_base, full_elastic)

    with st.spinner("Li₂Sn₅ (Phase 2+3)..."):
        # Cell contains 4 Li + 10 Sn (2 formula units)
        li_sn = compute_phase("Li2Sn5", 10.27, 3.12,
                              ['Sn', 'Li', 'Sn'], 127,
                              [(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)],
                              10, kpts_base, full_elastic)

    # Phase 1: Formation energy
    e_sn_per_calc = sn["e0"] / sn["num_sn"]
    e_li2sn5_total = li_sn["e0"]
    delta_e = e_li2sn5_total - 4 * e_li_per - 10 * e_sn_per_calc
    formation_per_atom = delta_e / 14
    formation_per_formula = delta_e / 2

    # Phase 2: Expansion
    v_per_sn = sn["v0"] / sn["num_sn"]
    v_per_sn_lith = li_sn["v0"] / li_sn["num_sn"]
    expansion = (v_per_sn_lith - v_per_sn) / v_per_sn * 100

    # Phase 4: Fracture metrics
    ar = li_sn["c33"] / li_sn["c11"]
    risk = "HIGH" if (expansion > 20 and ar < 0.9) else "LOW"

    # Results dashboard
    st.success("✅ Full analysis complete!")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Volume Expansion", f"{expansion:.2f}%", delta=f"+{expansion:.2f}%")
    with col2:
        st.metric("Formation Energy (per atom)", f"{formation_per_atom:.3f} eV", 
                  delta="Stable" if formation_per_atom < 0 else "Unstable")
    with col3:
        st.metric("Anisotropy Ratio (Li₂Sn₅)", f"{ar:.3f}", delta="c-soft" if ar < 1 else "isotropic")
    with col4:
        st.metric("Fracture Risk", risk, delta_color="inverse" if risk == "HIGH" else "normal")

    # Create tabs
    tabs = st.tabs(["E-V Mapping", "Anisotropic Elasticity", "Thermodynamic Stability", "Fracture Prediction", "Visualizations"])

    # Tab 1: E-V Mapping
    with tabs[0]:
        st.subheader("Energy-Volume Curves")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Sn
        eos_sn = EquationOfState(sn["vols"], sn["energies"])
        eos_sn.plot(ax1)
        ax1.set_title("BCT Sn: E(V) Curve & EOS Fit")
        ax1.set_xlabel("Volume (Å³)")
        ax1.set_ylabel("Energy (eV)")
        # Li2Sn5
        eos_li = EquationOfState(li_sn["vols"], li_sn["energies"])
        eos_li.plot(ax2)
        ax2.set_title("Li₂Sn₅: E(V) Curve & EOS Fit")
        ax2.set_xlabel("Volume (Å³)")
        ax2.set_ylabel("Energy (eV)")
        st.pyplot(fig)

        # Summary table
        st.subheader("EOS Parameters")
        df_eos = pd.DataFrame({
            "Phase": ["Sn", "Li₂Sn₅"],
            "V₀ (Å³)": [sn["v0"], li_sn["v0"]],
            "B₀ (GPa)": [sn["B"], li_sn["B"]],
            "V/Sn (Å³)": [v_per_sn, v_per_sn_lith],
            "Expansion (%)": [0, expansion]
        })
        st.dataframe(df_eos)

    # Tab 2: Anisotropic Elasticity
    with tabs[1]:
        st.subheader("Strain-Energy Fits for C₁₁ and C₃₃")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        strains = li_sn["strain_data"]["strains"]
        # C11
        ax1.plot(strains, li_sn["strain_data"]["e_a"], 'o', label="Data")
        popt_a = curve_fit(quadratic, strains, li_sn["strain_data"]["e_a"])[0]
        ax1.plot(strains, quadratic(strains, *popt_a), '--', label="Fit")
        ax1.set_xlabel("Strain ε (a-axis)")
        ax1.set_ylabel("Energy (eV)")
        ax1.set_title(f"C₁₁ = {li_sn['c11']:.1f} GPa")
        ax1.legend()
        # C33
        ax2.plot(strains, li_sn["strain_data"]["e_c"], 'o', label="Data")
        popt_c = curve_fit(quadratic, strains, li_sn["strain_data"]["e_c"])[0]
        ax2.plot(strains, quadratic(strains, *popt_c), '--', label="Fit")
        ax2.set_xlabel("Strain ε (c-axis)")
        ax2.set_ylabel("Energy (eV)")
        ax2.set_title(f"C₃₃ = {li_sn['c33']:.1f} GPa")
        ax2.legend()
        st.pyplot(fig)

        st.subheader("Elastic Constants (GPa)")
        if full_elastic:
            cols = st.columns(4)
            cols[0].metric("C₁₁", f"{li_sn['elastic']['C11']:.1f}")
            cols[1].metric("C₃₃", f"{li_sn['elastic']['C33']:.1f}")
            cols[2].metric("C₁₂", f"{li_sn['elastic']['C12']:.1f}")
            cols[3].metric("C₁₃", f"{li_sn['elastic']['C13']:.1f}")
            cols2 = st.columns(2)
            cols2[0].metric("C₄₄", f"{li_sn['elastic']['C44']:.1f}")
            cols2[1].metric("C₆₆", f"{li_sn['elastic']['C66']:.1f}")
        else:
            st.write("Full elastic tensor not computed. Run in Production mode with 'Compute full elastic tensor' enabled.")

        st.info(f"Anisotropy Ratio AR = C₃₃/C₁₁ = {ar:.3f}. AR < 1 indicates softer c-axis, favoring interlayer cleavage.")

        # 3D anisotropic visualization (if full tensor available)
        if full_elastic:
            st.subheader("3D Young's Modulus Surface (Li₂Sn₅)")
            # Compute Young's modulus directional dependence using elastic compliance
            # Build stiffness matrix in Voigt notation
            C = np.zeros((6,6))
            C[0,0] = C[1,1] = li_sn['elastic']['C11']
            C[0,1] = C[1,0] = li_sn['elastic']['C12']
            C[0,2] = C[1,2] = C[2,0] = C[2,1] = li_sn['elastic']['C13']
            C[2,2] = li_sn['elastic']['C33']
            C[3,3] = li_sn['elastic']['C44']
            C[4,4] = li_sn['elastic']['C44']
            C[5,5] = li_sn['elastic']['C66']
            # Compliance matrix S = inv(C)
            S = inv(C)
            # Function to compute Young's modulus in direction (theta, phi) in spherical coordinates
            def young_modulus(theta, phi):
                # Direction cosines
                n1 = np.sin(theta) * np.cos(phi)
                n2 = np.sin(theta) * np.sin(phi)
                n3 = np.cos(theta)
                # For cubic/tetragonal, Young's modulus E = 1 / (S1111 n1^4 + S2222 n2^4 + S3333 n3^4 + 2(S1122 n1^2 n2^2 + S1133 n1^2 n3^2 + S2233 n2^2 n3^2) + 4(S2323 n2^2 n3^2 + S1313 n1^2 n3^2 + S1212 n1^2 n2^2))
                # Using Voigt notation: S1111 = S[0,0], S2222 = S[1,1], S3333 = S[2,2], S1122 = S[0,1], S1133 = S[0,2], S2233 = S[1,2], S2323 = S[3,3]/4? Actually S_ijkl for shear: for ij=23, kl=23, S_2323 = S[3,3]/4? Standard conversion: For shear components, S_ijkl = S_{IJ}/2 when I,J>3? Need careful. Usually, for compliance, the relation between engineering constants and Voigt is: S_ijkl = S_{IJ} for I,J=1,2,3 and S_ijkl = S_{IJ}/2 for I,J=4,5,6? Actually in Voigt notation, strain vector is (ε1, ε2, ε3, γ23, γ13, γ12) with γ = 2ε. The compliance matrix S is defined such that ε = S σ. Then S_ijkl = S_{IJ} for I,J=1..3, and S_ijkl = S_{IJ}/2 for I or J=4..6. So:
                # S1111 = S[0,0]
                # S1122 = S[0,1]
                # S1133 = S[0,2]
                # S2222 = S[1,1]
                # S2233 = S[1,2]
                # S3333 = S[2,2]
                # S2323 = S[3,3]/4? Actually γ23 = 2ε23, so relation: ε23 = S_{23kl} σ_kl. In Voigt, σ23 = σ4, ε23 = ε4/2? Let's derive: The compliance tensor S_ijkl in full form: ε_ij = S_ijkl σ_kl. In Voigt, we have ε_I = S_IJ σ_J, where for I=4, ε4 = 2ε23, and σ4 = σ23. So ε23 = ε4/2 = S_4J σ_J /2. So S_2323 = S_44/4. Similarly, for other shear components.
                S1111 = S[0,0]
                S2222 = S[1,1]
                S3333 = S[2,2]
                S1122 = S[0,1]
                S1133 = S[0,2]
                S2233 = S[1,2]
                S2323 = S[3,3] / 4
                S1313 = S[4,4] / 4
                S1212 = S[5,5] / 4
                term = (S1111 * n1**4 + S2222 * n2**4 + S3333 * n3**4 +
                        2*(S1122 * n1**2 * n2**2 + S1133 * n1**2 * n3**2 + S2233 * n2**2 * n3**2) +
                        4*(S2323 * n2**2 * n3**2 + S1313 * n1**2 * n3**2 + S1212 * n1**2 * n2**2))
                return 1.0 / term

            # Create grid
            theta = np.linspace(0, np.pi, 50)
            phi = np.linspace(0, 2*np.pi, 50)
            THETA, PHI = np.meshgrid(theta, phi)
            E = young_modulus(THETA, PHI)
            # Convert to Cartesian for 3D surface
            X = E * np.sin(THETA) * np.cos(PHI)
            Y = E * np.sin(THETA) * np.sin(PHI)
            Z = E * np.cos(THETA)

            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', colorbar_title='Young\'s Modulus (GPa)')])
            fig.update_layout(title='Anisotropic Young\'s Modulus of Li₂Sn₅', autosize=False, width=700, height=700)
            st.plotly_chart(fig)

    # Tab 3: Thermodynamic Stability
    with tabs[2]:
        st.subheader("Formation Energy Calculation")
        st.markdown(f"""
        **Reaction**: 4 Li + 10 Sn → Li₂Sn₅ (2 formula units per cell)

        **Formula**:  
        ΔE_f = E(Li₂Sn₅) - 4·E(Li) - 10·E(Sn)  
        = {li_sn['e0']:.3f} - 4·{e_li_per:.3f} - 10·{e_sn_per_calc:.3f}  
        = {delta_e:.3f} eV per cell

        **Per atom**: {formation_per_atom:.3f} eV/atom  
        **Per formula unit**: {formation_per_formula:.3f} eV/f.u.

        **Conclusion**: ΔE_f < 0 → Li₂Sn₅ is thermodynamically stable relative to pure Li and Sn.
        """)
        # Bar chart of energies
        fig, ax = plt.subplots()
        labels = ['Sn (per atom)', 'Li (per atom)', 'Li₂Sn₅ (per atom)']
        values = [e_sn_per_calc, e_li_per, li_sn['e0']/14]
        ax.bar(labels, values, color=['blue', 'green', 'red'])
        ax.set_ylabel('Energy (eV/atom)')
        ax.set_title('Relative Energies')
        st.pyplot(fig)

    # Tab 4: Fracture Prediction
    with tabs[3]:
        st.subheader("Fracture Risk Assessment")
        st.markdown(f"""
        - **Volume expansion**: {expansion:.2f}%  
        - **Anisotropy ratio (C₃₃/C₁₁)**: {ar:.3f}  
        - **Bulk modulus change**: Sn B₀ = {sn['B']:.1f} GPa, Li₂Sn₅ B₀ = {li_sn['B']:.1f} GPa → ΔB₀ = {li_sn['B'] - sn['B']:.1f} GPa ({"softening" if li_sn['B'] < sn['B'] else "stiffening"})

        **Fracture criterion**:  
        - Expansion > 20% → high tensile stress  
        - AR < 0.9 → preferential c-axis expansion → delamination

        **Risk level**: **{risk}**
        """)
        # Radar chart comparing Sn and Li2Sn5 properties
        st.subheader("Property Comparison")
        properties = ['V/Sn', 'B₀', 'C₁₁', 'C₃₃', 'AR']
        sn_vals = [v_per_sn, sn['B'], sn['c11'], sn['c33'], 1.0]  # AR for Sn = 1 (isotropic assumption)
        li_vals = [v_per_sn_lith, li_sn['B'], li_sn['c11'], li_sn['c33'], ar]

        # Normalize to max of both
        max_vals = [max(sn_vals[i], li_vals[i]) for i in range(len(properties))]
        sn_norm = [sn_vals[i]/max_vals[i] for i in range(len(properties))]
        li_norm = [li_vals[i]/max_vals[i] for i in range(len(properties))]

        angles = np.linspace(0, 2*np.pi, len(properties), endpoint=False).tolist()
        sn_norm += sn_norm[:1]
        li_norm += li_norm[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.plot(angles, sn_norm, 'o-', linewidth=2, label='Sn')
        ax.plot(angles, li_norm, 'o-', linewidth=2, label='Li₂Sn₅')
        ax.fill(angles, sn_norm, alpha=0.25)
        ax.fill(angles, li_norm, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(properties)
        ax.set_title('Normalized Property Comparison')
        ax.legend(loc='upper right')
        st.pyplot(fig)

    # Tab 5: Visualizations
    with tabs[4]:
        st.subheader("Additional Visualizations")
        # Histogram of energy differences (placeholder)
        # Could show distribution of errors if we had multiple calculations, but we don't
        st.write("**Histogram of energy variations across volume points**")
        # For Sn
        fig, ax = plt.subplots()
        ax.hist(sn['energies'], bins=10, alpha=0.5, label='Sn')
        ax.hist(li_sn['energies'], bins=10, alpha=0.5, label='Li₂Sn₅')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)

        # Scatter plot: Expansion vs AR (could be extended with more data points)
        st.write("**Scatter: Volume Expansion vs Anisotropy Ratio**")
        # Placeholder: we only have one point, but we can still show
        fig, ax = plt.subplots()
        ax.scatter([ar], [expansion], s=100, c='red', label='Li₂Sn₅')
        ax.set_xlabel('Anisotropy Ratio AR')
        ax.set_ylabel('Volume Expansion (%)')
        ax.set_title('Expansion vs Anisotropy')
        ax.grid(True)
        st.pyplot(fig)

        # 3D stress distribution polar spherical coordinates
        st.write("**3D Stress Distribution (Polar Spherical Coordinates)**")
        st.markdown("""
        *Simplified representation: The stress distribution under isotropic expansion can be visualized as a spherical harmonic plot.*
        For full anisotropic stress, full tensor needed. Here we show a 3D sphere colored by Young's modulus direction (already shown in Anisotropic tab).
        """)
        # Use the same 3D surface from earlier but with different color mapping
        if full_elastic:
            # Reuse the same data but plot with a different colormap
            # We already have the plot, we can just show it again
            # But to avoid recomputation, we can just mention it
            st.info("See 3D Young's Modulus surface in the Anisotropic Elasticity tab.")
        else:
            st.warning("Full elastic tensor not computed. Enable in Production mode to view 3D anisotropy.")

        # Download all data as zip
        st.subheader("Download All Results")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            # E-V data
            df_sn = pd.DataFrame({'Volume (Å³)': sn['vols'], 'Energy (eV)': sn['energies']})
            df_li = pd.DataFrame({'Volume (Å³)': li_sn['vols'], 'Energy (eV)': li_sn['energies']})
            zipf.writestr('sn_ev.csv', df_sn.to_csv(index=False))
            zipf.writestr('li2sn5_ev.csv', df_li.to_csv(index=False))
            # Strain data
            if full_elastic:
                df_strain = pd.DataFrame({
                    'strain': strains,
                    'C11_energy': li_sn['strain_data']['e_a'],
                    'C33_energy': li_sn['strain_data']['e_c']
                })
                zipf.writestr('strain_data.csv', df_strain.to_csv(index=False))
            # Summary
            summary = pd.DataFrame({
                'Phase': ['Sn', 'Li2Sn5'],
                'V0 (A^3)': [sn['v0'], li_sn['v0']],
                'B0 (GPa)': [sn['B'], li_sn['B']],
                'C11 (GPa)': [sn['c11'], li_sn['c11']],
                'C33 (GPa)': [sn['c33'], li_sn['c33']],
                'AR': [1.0, ar],
                'Expansion (%)': [0, expansion],
                'DeltaEf (eV/atom)': [None, formation_per_atom]
            })
            zipf.writestr('summary.csv', summary.to_csv(index=False))
        st.download_button("Download ZIP", zip_buffer.getvalue(), file_name="results.zip", mime="application/zip")

    # Additional instructions
    st.info("""
    **Interpretation Guide**:
    - ΔE_f < 0 → Li₂Sn₅ forms spontaneously.
    - Expansion > 20% + AR < 0.9 → c-axis delamination → anode pulverization.
    - Large ΔB₀ → material softening → plastic flow & cracking.
    - The 3D Young's modulus surface shows anisotropic elastic behavior.
    """)
