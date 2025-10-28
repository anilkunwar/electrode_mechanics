import numpy as np
from pyscf.pbc import gto, dft
from pyscf.pbc.tools import lattice
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the Birch-Murnaghan equation of state
def birch_murnaghan(V, E0, V0, B0, Bp):
    eta = (V0 / V)**(2/3)
    return E0 + (9 * V0 * B0 / 16) * (
        (eta - 1)**3 * Bp + (eta - 1)**2 * (6 - 4 * eta)
    )

# Example: Compute E(V) curve for BCT Sn (beta-Sn, tetragonal)
# Lattice parameters: a = 5.83 Å, c = 3.18 Å (initial guess)
# Spacegroup 141 (I41/amd), but for simplicity, define cell manually

# Define the cell (tetragonal, 4 Sn atoms)
atom = 'Sn 0 0 0; Sn 0.5 0.5 0.5; Sn 0.0 0.5 0.25; Sn 0.5 0.0 0.75'  # Positions for BCT Sn
a_init = 5.83
c_init = 3.18
cell_init = np.array([
    [a_init, 0, 0],
    [0, a_init, 0],
    [0, 0, c_init]
])

# Volume scales (e.g., 90% to 110% of initial volume)
v_init = np.linalg.det(cell_init)  # Initial volume
scale_factors = np.linspace(0.9, 1.1, 11)  # 11 points from 0.9 to 1.1
volumes = v_init * scale_factors
energies = []

# DFT settings: Use PBE functional, small basis for demo (in practice, use better basis/kpoints)
basis = 'sto-3g'  # Minimal basis for speed; use def2-svp or better in production
kpts = [2, 2, 3]  # Small k-grid for demo; increase for accuracy

for scale in scale_factors:
    # Scale the cell isotropically (for tetragonal, scale a and c proportionally to keep c/a ratio)
    ratio = c_init / a_init
    a_scaled = a_init * scale**(1/3)
    c_scaled = a_scaled * ratio  # Preserve aspect ratio for simplicity
    cell_scaled = np.array([
        [a_scaled, 0, 0],
        [0, a_scaled, 0],
        [0, 0, c_scaled]
    ])
    
    # Build the cell
    cell = gto.Cell()
    cell.atom = atom
    cell.a = cell_scaled
    cell.basis = basis
    cell.unit = 'A'
    cell.build()
    
    # DFT calculation (relax positions at fixed volume? For full vc-relax, but here fixed cell)
    # For E(V), often relax ions at each V
    mf = dft.RKS(cell)
    mf.xc = 'pbe'
    mf.kpts = cell.make_kpts(kpts)
    mf.kernel()  # Compute energy
    
    energy = mf.e_tot  # Total energy per unit cell
    energies.append(energy)

# Normalize energies (subtract min for fitting)
energies = np.array(energies)
energies -= np.min(energies)

# Fit to Birch-Murnaghan EOS
popt, pcov = curve_fit(birch_murnaghan, volumes, energies, p0=[0, v_init, 100, 4])  # Initial guesses: E0=0, V0=v_init, B0=100 GPa, Bp=4
E0, V0, B0, Bp = popt

print(f"Equilibrium volume V0: {V0:.4f} Å³ (per unit cell)")
print(f"Bulk modulus B0: {B0:.4f} (in atomic units; convert to GPa as needed)")

# Plot E(V) curve
plt.plot(volumes, energies, 'o', label='DFT points')
v_fit = np.linspace(min(volumes), max(volumes), 100)
plt.plot(v_fit, birch_murnaghan(v_fit, *popt), '-', label='Fit')
plt.xlabel('Volume (Å³)')
plt.ylabel('Relative Energy (Ha)')
plt.legend()
plt.show()

# For volume expansion: Repeat for Li2Sn5, normalize per Sn atom, compute relative change
# (Similar setup, but with Li2Sn5 structure: spacegroup 127, etc.)
