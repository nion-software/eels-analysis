import numpy
import EELS_DataAnalysis as analyzer

# Load and reformat DM data array
eels_spectra = numpy.fromfile("test/MagiCal sample data/EELS Spectrum Image.dat", numpy.float32)
eels_spectra.resize(2048, 30, 150)
eels_spectra = numpy.swapaxes(eels_spectra, 0, 2)
eels_spectra = numpy.swapaxes(eels_spectra, 0, 1)
spectral_range = numpy.array([418.92, 2405.48])

edge_delta = 200.0

# Extract and output Si K edge
edge_onset = 1839.0
bkgd_range = numpy.array([1700.0, 1800.0])
edge_map, edge_profile, bkgd_model, profile_range = analyzer.core_loss_edge(eels_spectra, spectral_range, edge_onset, edge_delta, bkgd_range)
edge_map.tofile("test/MagiCal/Si K edge map.dat")
edge_profile = numpy.swapaxes(edge_profile, 0, 1)
edge_profile = numpy.swapaxes(edge_profile, 0, 2)
edge_profile.tofile("test/MagiCal/Si K edge profile.dat")
bkgd_model = numpy.swapaxes(bkgd_model, 0, 1)
bkgd_model = numpy.swapaxes(bkgd_model, 0, 2)
bkgd_model.tofile("test/MagiCal/Si K background.dat")
print("Si K range", profile_range)

# Extract and output Ge L edge
edge_onset = 1220.0
bkgd_range = numpy.array([1100.0, 1200.0])
edge_map, edge_profile, bkgd_model, profile_range = analyzer.core_loss_edge(eels_spectra, spectral_range, edge_onset, edge_delta, bkgd_range)
edge_map.tofile("test/MagiCal/Ge L edge map.dat")
edge_profile = numpy.swapaxes(edge_profile, 0, 1)
edge_profile = numpy.swapaxes(edge_profile, 0, 2)
edge_profile.tofile("test/MagiCal/Ge L edge profile.dat")
bkgd_model = numpy.swapaxes(bkgd_model, 0, 1)
bkgd_model = numpy.swapaxes(bkgd_model, 0, 2)
bkgd_model.tofile("test/MagiCal/Ge L background.dat")
print("Ge L range", profile_range)
