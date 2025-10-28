import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.spacegroup import crystal
from ase.constraints import ExpCellFilter
from ase.units import GPa
from ase.eos import EquationOfState
from gpaw import GPAW, PW
import matplotlib.pyplot as plt

# Function to compute energy at fixed volume (with ionic relaxation)
def get_energy_at_volume(atoms_template, volume, ecut=500, kpts=(8,8,12), xc='PBE'):
    # Scale cell to target volume while preserving shape ratios
    cell = atoms_template.get_cell()
    current_volume = atoms_template.get_volume()
    scale = (volume / current_volume) ** (1.0 / 3.0)
    new_cell = cell * scale
    atoms = atoms_template.copy()
    atoms.set_cell(new_cell, scale_atoms=True)
    
    # Set GPAW calculator
    calc = GPAW(
        mode=PW(ecut),
        xc=xc,
        kpts=kpts,
        txt='gpaw.log',
        convergence={'energy': 1e-5}
    )
    atoms.calc = calc
    
    # Relax ionic positions at fixed cell
    opt = BFGS(atoms, logfile='relax.log')
    opt.run(fmax=0.01)  # Loose convergence for speed; tighten for accuracy
    
    return atoms.get_potential_energy()

# Main script for E(V) curve and fit
def compute_eos(structure, volumes_rel=np.linspace(0.9, 1.1, 11), ecut=500, kpts=(8,8,12)):
    if structure == 'Sn (BCT)':
        a = 5.83
        c = 3.18
        atoms_template = crystal('Sn', basis=[(0,0,0)], spacegroup=141, cellpar=[a, a, c, 90, 90, 90])
        num_sn = len(atoms_template)  # 4
    elif structure == 'Li2Sn5':
        a = 10.274
        c = 3.125
        atoms_template = crystal(
            symbols=['Sn', 'Li', 'Sn'],
            basis=[(0, 0.5, 0), (0.16, 0.66, 0), (0.295, 0.432, 0)],
            spacegroup=127,
            cellpar=[a, a, c, 90, 90, 90]
        )
        num_sn = atoms_template.get_chemical_symbols().count('Sn')  # 10
    else:
        raise ValueError("Unsupported structure")

    v0 = atoms_template.get_volume()
    volumes = v0 * volumes_rel
    energies = []
    
    for v in volumes:
        e = get_energy_at_volume(atoms_template, v, ecut=ecut, kpts=kpts)
        energies.append(e)
    
    # Fit to EquationOfState (Birch-Murnaghan by default)
    eos = EquationOfState(volumes, energies)
    v_fit, e_fit, B_fit, Bp_fit = eos.fit()
    
    # B in eV/Å³; convert to GPa
    B_gpa = B_fit / GPa
    
    print(f"Equilibrium volume V0: {v_fit:.4f} Å³ (unit cell)")
    print(f"Volume per Sn: {v_fit / num_sn:.4f} Å³")
    print(f"Bulk modulus B: {B_gpa:.2f} GPa")
    
    # Plot
    eos.plot()
    plt.show()
    
    return v_fit / num_sn  # Return volume per Sn

# Compute for BCT Sn and Li2Sn5
v_sn = compute_eos('Sn (BCT)')
v_li2sn5 = compute_eos('Li2Sn5')

# Volume expansion
expansion = (v_li2sn5 - v_sn) / v_sn * 100
print(f"Volume expansion: {expansion:.2f}%")
