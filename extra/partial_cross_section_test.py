import numpy
import EELS_CrossSections as x_sect

atomic_number = 14
edge_onset_eV = 1839.0
edge_delta_eV = 200.0
beam_energy_eV = 2.0e5
alpha_rad = 0.0034
beta_rad = .019

x_section = x_sect.partial_cross_section_nm2(atomic_number, 1, 1, edge_onset_eV, edge_delta_eV, beam_energy_eV, alpha_rad, beta_rad)
print("Z =", atomic_number)
print("Ek =", edge_onset_eV)
print("D =", edge_delta_eV)
print("E0 =", beam_energy_eV)
print("A =", alpha_rad)
print("B =", beta_rad)
print("X-section =", x_section, "nm**2")
