import numpy
import EELS_DataAnalysis as analyzer

# Load and reformat DM data array
eels_spectra = numpy.fromfile("test/MagiCal sample data/EELS Spectrum Image.dat", numpy.float32)
eels_spectra.resize(2048, 30, 150)
eels_spectra = numpy.swapaxes(eels_spectra, 0, 2)
eels_spectra = numpy.swapaxes(eels_spectra, 0, 1)
spectral_range = numpy.array([418.92, 2405.48])

edge_delta = 200.0

# Extract and output Si abundance
edge_onset = 1839.0
bkgd_range = numpy.array([1700.0, 1800.0])
beam_energy_eV = 2.0e5
alpha_rad = 0.0034
beta_rad = .019

abundance_map = analyzer.relative_atomic_abundance(eels_spectra, spectral_range, bkgd_range, 14, edge_onset, edge_delta, beam_energy_eV, alpha_rad, beta_rad)
abundance_map.tofile("test/MagiCal/Si abundance.dat")
