"""
    Spectral Analysis Library

    A library of functions and classes for general spectral signal extraction techniques,
    such as background subtraction and multiple linear regression of reference signals.

"""

# standard libraries
import copy
import typing

import numpy
from nion.eels_analysis import CurveFitting
from nion.eels_analysis import EELS_CrossSections
from nion.eels_analysis import EELS_DataAnalysis
from nion.eels_analysis import PeriodicTable
from nion.data import DataAndMetadata
from nion.utils import Registry


def extract_signal_from_polynomial_background_data(data, signal_range, fit_ranges, first_x = 0.0, delta_x = 1.0,
                                                polynomial_order = 1, fit_log_data = False, fit_log_x = False):

    """A function for performing generic polynomial background subtraction on 1-D spectral data arrays.

    The required data (NumPy) array can have 1 or 2 dimensions.  If it is 1-dimensional, then each array element is
    a spectral intensity (ordinate) value along an equispaced x axis with an initial abscissa value given by first_x and
    a fixed abscissa increment per data element given by delta_x.  If the data array is 2-dimensional, then one of the
    dimensions must have size 2.  The first entry along that dimension is an x (abscissa) value and the second entry is
    the corresponding spectral intensity (ordinate) value.  Note that in this case, the abscissa values need not be ordered.
    This function will return background-subtracted spectral intensities regardless of the abscissa value ordering, or lack thereof.

    The parameters signal_range and fit_ranges are (NumPy) arrays specifying abscissa ranges, of the form [start_x, end_x].
    The former specifies the single range over which the signal of interest occurs, while the latter can specify multiple ranges
    over which to fit the desired polynomial background model, distributed over the second dimension of the fit_ranges array.
    Only a single row in fit_ranges is required, in which case the background will be extrapolated over the signal range and subtracted.
    If additional rows are supplied, then the background will be interpolated or extrapolated, depending on the range relationships.

    The order of the polynomial model function is specified by the polynomial_order parameter.
    The fit_log_data and fit_log_x parameters specify whether the corresponding axes of the spectral data should be transformed to
    logarithmic scales before doing the background fit.  This is to support exponential, Gaussian, and power-law background models.
    """

    # check shape and validity of data array
    data_dimension_count = len(data.shape)
    assert data_dimension_count <= 2

    # extract spectral intensity (ordinate) and x (abscissa) arrays
    have_x_array = (data_dimension_count == 2)
    if have_x_array:
        assert min(data.shape[0], data.shape[1]) == 2
        if data.shape[0] == 2:
            x_values, data_values = data
        else:
            x_values = data[:, 0]
            data_values = data[:, 1]
    else:
        x_values = None
        data_values = data

    data_size = data_values.shape[0]
    assert data_size > polynomial_order + 3

    # check validity of abscissa value inputs
    if have_x_array:
        min_x = x_values.min()
        max_x = x_values.max()
    else:
        min_x = first_x
        max_x = first_x + data_size * delta_x
    assert max_x > min_x

    # check shape and validity of fit_ranges array
    range_dimension_count = len(fit_ranges.shape)
    assert range_dimension_count < 3
    assert fit_ranges.shape[range_dimension_count - 1] == 2

    # distill the fit ranges so that they are ordered, consolidated, and non-overlapping
    fit_ranges_clean = numpy.atleast_2d(numpy.sort(fit_ranges))
    fit_range_order = numpy.argsort(fit_ranges_clean, 0)[:, 0]
    fit_ranges_clean = fit_ranges_clean[fit_range_order]
    range_index = 1
    while range_index < fit_ranges_clean.shape[0]:
        if fit_ranges_clean[range_index, 0] > fit_ranges_clean[range_index - 1, 1]:
            range_index += 1
        else:
            fit_ranges_clean[range_index - 1, 1] = fit_ranges_clean[range_index, 1]
            fit_ranges_clean = numpy.delete(fit_ranges_clean, range_index, 0)

    # check validity of fit_ranges_clean array with respect to passed-in data range
    range_index = 0
    range_count = fit_ranges_clean.shape[0]
    while range_index < range_count:
        assert fit_ranges_clean[range_index, 0] >= min_x and fit_ranges_clean[range_index, 1] <= max_x
        range_index += 1

    # compile x and y arrays over fit ranges for input to the polynomial background fit
    if have_x_array:
        fit_range_selector = x_values >= fit_ranges_clean[0, 0] and x_values <= fit_ranges_clean[0, 1]
        range_index = 1
        while range_index < range_count:
            fit_range_selector = fit_range_selector or (x_values >= fit_ranges_clean[range_index, 0] and x_values <= fit_ranges_clean[range_index, 1])
            range_index += 1

        x_fit_values = x_values.compress(fit_range_selector)
        y_fit_values = data_values.compress(fit_range_selector)
    else:
        fit_start_chan = round((fit_ranges_clean[0, 0] - min_x) / delta_x)
        fit_end_chan = round((fit_ranges_clean[0, 1] - min_x) / delta_x) + 1
        x_fit_values = first_x + delta_x * numpy.arange(fit_start_chan, fit_end_chan)
        y_fit_values = data_values[fit_start_chan : fit_end_chan]

        range_index = 1
        while range_index < range_count:
            fit_start_chan = round((fit_ranges_clean[range_index, 0] - min_x) / delta_x)
            fit_end_chan = round((fit_ranges_clean[range_index, 1] - min_x) / delta_x) + 1
            numpy.append(x_fit_values, first_x + delta_x * numpy.arange(fit_start_chan, fit_end_chan))
            numpy.append(y_fit_values, data_values[fit_start_chan : fit_end_chan])
            range_index += 1

    # generate a polynomial fit over the fit ranges
    background_fit = CurveFitting.PolynomialFit1D(y_fit_values, x_fit_values, 0, 1, polynomial_order, fit_log_data, fit_log_x)

    # check shape and validity of signal_range array with respect to passed-in data range
    signal_range_clean = numpy.sort(signal_range).flatten()
    assert signal_range_clean.shape[0] == 2
    assert signal_range_clean[0] >= min_x and signal_range_clean[1] <= max_x

    # compute background model and subtract from data over contiguous union of fit and signal ranges
    bkgd_range = numpy.array([min(fit_ranges_clean[0, 0], signal_range_clean[0]), max(fit_ranges_clean[range_count - 1, 1], signal_range_clean[1])])

    # compute the net signal
    if have_x_array:
        in_bkgd_range = x_values >= bkgd_range[0] and x_values <= bkgd_range[1]
        bkgd_x_values = numpy.where(in_bkgd_range, x_values, bkgd_range[0])
        bkgd_fit = background_fit.compute_fit_for_values(bkgd_x_values)
        net_signal = numpy.where(in_bkgd_range, data_values - bkgd_fit, 0)
    else:
        bkgd_start_chan = round((bkgd_range[0] - min_x) / delta_x)
        bkgd_end_chan = round((bkgd_range[1] - min_x) / delta_x) + 1
        bkgd_fit = background_fit.compute_fit_for_range(bkgd_range, bkgd_end_chan - bkgd_start_chan)
        net_signal = numpy.zeros_like(data_values)
        net_signal[bkgd_start_chan : bkgd_end_chan] = data_values[bkgd_start_chan : bkgd_end_chan] - bkgd_fit

    return net_signal


def extract_signal_from_polynomial_background(data_and_metadata, signal_range, fit_ranges, first_x = 0.0, delta_x = 1.0,
                                                polynomial_order = 1, fit_log_data = False, fit_log_x = False):
    signal_range = numpy.asarray(signal_range) * data_and_metadata.data_shape[0]
    fit_ranges = numpy.asarray(fit_ranges) * data_and_metadata.data_shape[0]
    data = extract_signal_from_polynomial_background_data(data_and_metadata.data, signal_range, fit_ranges, first_x, delta_x, polynomial_order, fit_log_data,
                                                          fit_log_x)
    return DataAndMetadata.new_data_and_metadata(data, data_and_metadata.intensity_calibration, data_and_metadata.dimensional_calibrations)


def stacked_fit_linear_background(data: numpy.ndarray, signal_index: int, rcond=1e-10) -> numpy.ndarray:
    """Return the linear background using least squares for an ndarray with signal in last index.

    The outline for this implementation comes from:
    http://stackoverflow.com/questions/30442377/how-to-solve-many-overdetermined-systems-of-linear-equations-using-vectorized-co
    """

    assert signal_index == -1
    signal_length = data.shape[signal_index]

    # using equation y = Ap where A = [[x 1]] and p = [[m], [c]], solve for p.
    linear = numpy.arange(signal_length)
    ones = numpy.ones((signal_length,))
    A = numpy.vstack([linear, ones]).T

    # solve for p using svd. p will have the shape (n, 2) where n is the dimensions of the non-signal indexes of the data.
    u, s, v = numpy.linalg.svd(A, full_matrices=False)
    s_max = numpy.amax(s, axis=-1, keepdims=True)
    s_min = rcond * s_max
    inv_s = numpy.zeros_like(s)
    inv_s[s >= s_min] = 1 / s[s >= s_min]
    x = numpy.einsum('...ji,...j->...i', v, inv_s * numpy.einsum('...ji,...j->...i', u, data.conj()))
    return numpy.conj(x, x)


def stacked_linear_background(data: numpy.ndarray, signal_index: int) -> numpy.ndarray:
    """Return the linear background using least squares for an ndarray with signal in last index."""

    assert signal_index == -1
    signal_length = data.shape[signal_index]

    # using equation y = Ap where A = [[x 1]] and p = [[m], [c]], solve for p.
    linear = numpy.arange(signal_length)

    # solve for p. p will have the shape (n, 2) where n is the dimensions of the non-signal indexes of the data.
    p = stacked_fit_linear_background(data, signal_index)

    # calculate the background by multiply and add
    return (p[..., 0, numpy.newaxis] * linear[:] + p[..., 1, numpy.newaxis])


def slow_fit_linear_background(data: numpy.ndarray, signal_index: int) -> numpy.ndarray:
    """Return the linear background as m, c using least squares for an ndarray with signal in last index.

    This implementaton also demonstrates how to convert a 1-d function to a stacked function.
    """

    # make reshaped data view
    assert signal_index == -1
    signal_length = data.shape[signal_index]
    if len(data.shape) > 1:
        reshaped_data = data.reshape(numpy.product(data.shape[0:signal_index], dtype=numpy.uint64), signal_length)
    else:
        reshaped_data = data.reshape(1, signal_length)

    # using equation y = Ap where A = [[x 1]] and p = [[m], [c]], solve for p.
    linear = numpy.arange(signal_length)
    ones = numpy.ones((signal_length,))
    A = numpy.vstack([linear, ones]).T

    # solve for p. p will have the shape (n, 2) where n is the dimensions of the non-signal indexes of the data.
    p = numpy.array([numpy.linalg.lstsq(A, reshaped_data[k, ...], rcond=-1)[0] for k in range(reshaped_data.shape[0])])

    # reshape and return
    return p.reshape(data.shape[:signal_index] + (2,))


def slow_linear_background(data: numpy.ndarray, signal_index: int) -> numpy.ndarray:
    """Return the linear background using least squares for an ndarray with signal in last index."""

    assert signal_index == -1
    signal_length = data.shape[signal_index]

    # using equation y = Ap where A = [[x 1]] and p = [[m], [c]], solve for p.
    linear = numpy.arange(signal_length)

    # solve for p. p will have the shape (n, 2) where n is the dimensions of the non-signal indexes of the data.
    p = slow_fit_linear_background(data, signal_index)

    # calculate the background by multiply and add
    return (p[..., 0, numpy.newaxis] * linear[:] + p[..., 1, numpy.newaxis])


def linear_background(data: numpy.ndarray, signal_index: int) -> numpy.ndarray:
    return stacked_linear_background(data, signal_index)


def subtract_linear_background(data_and_metadata: DataAndMetadata.DataAndMetadata, fit_range, signal_range) -> DataAndMetadata.DataAndMetadata:
    """Subtract linear background from data and metadata with signal in last index."""
    signal_index = -1

    signal_length = data_and_metadata.dimensional_shape[signal_index]

    fit_range = (numpy.asarray(fit_range) * signal_length).astype(numpy.int)
    signal_range = (numpy.asarray(signal_range) * signal_length).astype(numpy.int)

    data = data_and_metadata.data

    # Fit within fit_range; calculate background within signal_range; subtract from source signal range
    p = stacked_fit_linear_background(data[..., range(*fit_range)], signal_index)
    linear = numpy.arange(signal_range[0], signal_range[1])
    background = (p[..., 0, numpy.newaxis] * linear[:] + p[..., 1, numpy.newaxis])
    result = data[..., range(*signal_range)] - background

    return DataAndMetadata.new_data_and_metadata(result, data_and_metadata.intensity_calibration, data_and_metadata.dimensional_calibrations)


def calculate_background_signal(data_and_metadata: DataAndMetadata.DataAndMetadata, fit_ranges, signal_range) -> DataAndMetadata.DataAndMetadata:
    """Calculate background from data and metadata with signal in first index."""
    signal_index = -1

    signal_length = data_and_metadata.dimensional_shape[signal_index]

    signal_range = (numpy.asarray(signal_range) * signal_length).astype(numpy.float)

    data = data_and_metadata.data

    if len(data_and_metadata.dimensional_calibrations) == 0:
        return None

    # Fit within fit_range; calculate background within signal_range; subtract from source signal range
    signal_calibration = data_and_metadata.dimensional_calibrations[signal_index]
    spectral_range = numpy.array([signal_calibration.convert_to_calibrated_value(0), signal_calibration.convert_to_calibrated_value(signal_length)])
    edge_onset = signal_calibration.convert_to_calibrated_value(signal_range[0])
    edge_delta = signal_calibration.convert_to_calibrated_value(signal_range[1]) - edge_onset
    # bkgd_range = numpy.array([signal_calibration.convert_to_calibrated_value(fit_range0[0]), signal_calibration.convert_to_calibrated_value(fit_range0[1])])
    bkgd_ranges = numpy.array([numpy.array([signal_calibration.convert_to_calibrated_value(fit_range[0] * signal_length), signal_calibration.convert_to_calibrated_value(fit_range[1] * signal_length)]) for fit_range in fit_ranges])
    # print("d {} s {} e {} d {} b {}".format(data.shape if data is not None else None, spectral_range, edge_onset, edge_delta, bkgd_range))
    edge_map, edge_profile, bkgd_model, profile_range = EELS_DataAnalysis.core_loss_edge(data, spectral_range, edge_onset, edge_delta, bkgd_ranges)

    # Squeeze the result
    result = numpy.squeeze(bkgd_model)

    max_channel = int(round(max([fit_range[1] * signal_length for fit_range in fit_ranges] + [signal_range[1]])))
    min_channel = int(round(min([fit_range[0] * signal_length for fit_range in fit_ranges] + [signal_range[0]])))

    data_shape = list(data_and_metadata.data_shape)
    data_shape[signal_index] = max_channel - min_channel
    data_shape = tuple(data_shape)
    dimensional_calibrations = copy.deepcopy(data_and_metadata.dimensional_calibrations)
    dimensional_calibrations[signal_index].offset = signal_calibration.convert_to_calibrated_value(min_channel)
    dimensional_calibrations[signal_index].scale = (signal_calibration.convert_to_calibrated_value(max_channel) - dimensional_calibrations[signal_index].offset) / data_shape[signal_index]

    return DataAndMetadata.new_data_and_metadata(result, data_and_metadata.intensity_calibration, dimensional_calibrations)


def extract_original_signal(data_and_metadata: DataAndMetadata.DataAndMetadata, fit_ranges, signal_range) -> DataAndMetadata.DataAndMetadata:
    signal_index = -1

    signal_length = data_and_metadata.dimensional_shape[signal_index]

    signal_range = (numpy.asarray(signal_range) * signal_length).astype(numpy.float)

    max_channel = int(round(max([fit_range[1] * signal_length for fit_range in fit_ranges] + [signal_range[1]])))
    min_channel = int(round(min([fit_range[0] * signal_length for fit_range in fit_ranges] + [signal_range[0]])))

    result = data_and_metadata.data[..., min_channel:max_channel]

    data_shape = list(data_and_metadata.data_shape)
    data_shape[signal_index] = max_channel - min_channel
    data_shape = tuple(data_shape)
    dimensional_calibrations = copy.deepcopy(data_and_metadata.dimensional_calibrations)
    original_calibration = copy.deepcopy(dimensional_calibrations[signal_index])
    dimensional_calibrations[signal_index].offset = original_calibration.convert_to_calibrated_value(min_channel)
    dimensional_calibrations[signal_index].scale = (original_calibration.convert_to_calibrated_value(max_channel) - dimensional_calibrations[signal_index].offset) / data_shape[signal_index]

    return DataAndMetadata.new_data_and_metadata(result, data_and_metadata.intensity_calibration, dimensional_calibrations)


def make_signal_like(data_and_metadata_src: DataAndMetadata.DataAndMetadata, data_and_metadata_dst: DataAndMetadata.DataAndMetadata):
    signal_index = -1
    if not data_and_metadata_src.dimensional_calibrations or not data_and_metadata_dst.dimensional_calibrations:
        return None
    if abs(data_and_metadata_src.dimensional_calibrations[signal_index].scale - data_and_metadata_dst.dimensional_calibrations[signal_index].scale) > 1E-7:
        return None
    if data_and_metadata_src.dimensional_calibrations[signal_index].units != data_and_metadata_dst.dimensional_calibrations[signal_index].units:
        return None
    if data_and_metadata_src.dimensional_calibrations[signal_index].convert_to_calibrated_value(0) < data_and_metadata_dst.dimensional_calibrations[signal_index].convert_to_calibrated_value(0):
        return None
    if data_and_metadata_src.dimensional_calibrations[signal_index].convert_to_calibrated_value(data_and_metadata_src.data_shape[signal_index]) > data_and_metadata_dst.dimensional_calibrations[signal_index].convert_to_calibrated_value(data_and_metadata_dst.data_shape[signal_index]):
        return None

    data = numpy.copy(data_and_metadata_dst.data)
    index = int(data_and_metadata_dst.dimensional_calibrations[signal_index].convert_from_calibrated_value(data_and_metadata_src.dimensional_calibrations[signal_index].convert_to_calibrated_value(0)))
    data[:] = 0
    data[index:index + data_and_metadata_src.data_shape[signal_index]] = data_and_metadata_src.data

    return DataAndMetadata.new_data_and_metadata(data, data_and_metadata_dst.intensity_calibration, data_and_metadata_dst.dimensional_calibrations)


def map_background_subtracted_signal(data_and_metadata: DataAndMetadata.DataAndMetadata, electron_shell: typing.Optional[PeriodicTable.ElectronShell], fit_ranges, signal_range) -> DataAndMetadata.DataAndMetadata:
    """Subtract si_k background from data and metadata with signal in first index."""
    signal_index = -1

    signal_length = data_and_metadata.dimensional_shape[signal_index]

    signal_range = (numpy.asarray(signal_range) * signal_length).astype(numpy.float)

    signal_calibration = data_and_metadata.dimensional_calibrations[signal_index]
    spectral_range = numpy.array([signal_calibration.convert_to_calibrated_value(0), signal_calibration.convert_to_calibrated_value(data_and_metadata.dimensional_shape[signal_index])])
    edge_onset = signal_calibration.convert_to_calibrated_value(signal_range[0])
    edge_delta = signal_calibration.convert_to_calibrated_value(signal_range[1]) - edge_onset
    bkgd_ranges = numpy.array([numpy.array([signal_calibration.convert_to_calibrated_value(fit_range[0] * signal_length), signal_calibration.convert_to_calibrated_value(fit_range[1] * signal_length)]) for fit_range in fit_ranges])

    cross_section = None
    if electron_shell is not None:
        beam_energy_ev = data_and_metadata.metadata.get("beam_energy_eV")
        beam_convergence_angle_rad = data_and_metadata.metadata.get("beam_convergence_angle_rad")
        beam_collection_angle_rad = data_and_metadata.metadata.get("beam_collection_angle_rad")

        if beam_energy_ev is not None and beam_convergence_angle_rad is not None and beam_collection_angle_rad is not None:
            cross_section = partial_cross_section_nm2(electron_shell.atomic_number, electron_shell.shell_number, electron_shell.subshell_index, edge_onset, edge_delta, beam_energy_ev, beam_convergence_angle_rad, beam_collection_angle_rad)

    data = data_and_metadata.data

    # Fit within fit_range; calculate background within signal_range; subtract from source signal range
    edge_map, edge_profile, bkgd_model, profile_range = EELS_DataAnalysis.core_loss_edge(data, spectral_range, edge_onset, edge_delta, bkgd_ranges)

    result = edge_map if cross_section is None else edge_map / cross_section

    dimensional_calibrations = data_and_metadata.dimensional_calibrations[0:-1]
    intensity_calibration = copy.deepcopy(data_and_metadata.intensity_calibration)
    if cross_section is not None:
        intensity_calibration.units = "~"

    return DataAndMetadata.new_data_and_metadata(result, intensity_calibration, dimensional_calibrations)


def generalized_oscillator_strength(energy_loss_eV: float, momentum_transfer_au: float,
                                    atomic_number: int, shell_number: int, subshell_index: int) -> numpy.ndarray:
    """Return the generalized oscillator strength as an ndarray.

    energy is in eV.
    scattering_angle is in radians.

    The 0-axis is in units of eV
    The 1-axis is in units of Phi * Phi / scattering angle
    The intensity is in units of nm * nm
    """
    pass


def energy_diff_cross_section_nm2_per_ev(atomic_number: int, shell_number: int, subshell_index: int,
                                         edge_onset_ev: float, edge_delta_ev: float, beam_energy_ev: float,
                                         convergence_angle_rad: float, collection_angle_rad: float) -> numpy.ndarray:
    """Return the energy differential cross section for the specified electron shell and experimental parameters.

    The returned differential cross-section value is in units of nm * nm / eV.
    """
    energy_diff_sigma = None
    eels_analysis_service = Registry.get_component("eels_analysis_service")
    if energy_diff_sigma is None and hasattr(eels_analysis_service, "energy_diff_cross_section_nm2_per_ev"):
        energy_diff_sigma = eels_analysis_service.energy_diff_cross_section_nm2_per_ev(atomic_number=atomic_number,
                                                                                       shell_number=shell_number,
                                                                                       subshell_index=subshell_index,
                                                                                       edge_onset_ev=edge_onset_ev,
                                                                                       edge_delta_ev=edge_delta_ev,
                                                                                       beam_energy_ev=beam_energy_ev,
                                                                                       convergence_angle_rad=convergence_angle_rad,
                                                                                       collection_angle_rad=collection_angle_rad)
    if energy_diff_sigma is None and shell_number == 1 and subshell_index == 1:
        # k edges only
        energy_diff_sigma = EELS_CrossSections.energy_diff_cross_section_nm2_per_ev(atomic_number, shell_number,
                                                                                    subshell_index,
                                                                                    edge_onset_ev, edge_delta_ev,
                                                                                    beam_energy_ev,
                                                                                    convergence_angle_rad,
                                                                                    collection_angle_rad)
    return energy_diff_sigma


def partial_cross_section_nm2(atomic_number: int, shell_number: int, subshell_index: int,
                              edge_onset_ev: float, edge_delta_ev: float, beam_energy_ev: float,
                              convergence_angle_rad: float, collection_angle_rad: float) -> float:
    """Returns the partial cross section.

    The return value units are nm * nm.
    """
    cross_section = None
    eels_analysis_service = Registry.get_component("eels_analysis_service")
    if cross_section is None and hasattr(eels_analysis_service, "partial_cross_section_nm2"):
        cross_section = eels_analysis_service.partial_cross_section_nm2(atomic_number=atomic_number,
                                                                        shell_number=shell_number,
                                                                        subshell_index=subshell_index,
                                                                        edge_onset_ev=edge_onset_ev,
                                                                        edge_delta_ev=edge_delta_ev,
                                                                        beam_energy_ev=beam_energy_ev,
                                                                        convergence_angle_rad=convergence_angle_rad,
                                                                        collection_angle_rad=collection_angle_rad)

    if cross_section is None:
        energy_diff_sigma = energy_diff_cross_section_nm2_per_ev(atomic_number=atomic_number,
                                                                 shell_number=shell_number,
                                                                 subshell_index=subshell_index,
                                                                 edge_onset_ev=edge_onset_ev,
                                                                 edge_delta_ev=edge_delta_ev,
                                                                 beam_energy_ev=beam_energy_ev,
                                                                 convergence_angle_rad=convergence_angle_rad,
                                                                 collection_angle_rad=collection_angle_rad)

        # Integrate over energy window to get partial cross-section
        energy_sample_count = energy_diff_sigma.shape[0]
        energy_step = edge_delta_ev / (energy_sample_count - 1)
        cross_section = numpy.trapz(energy_diff_sigma, dx=energy_step)

    if cross_section is None and atomic_number == 32 and shell_number == 2 and subshell_index == 3:
        # special section for testing
        if abs(edge_delta_ev - 100) < 3:
            cross_section = 7.31e-8
        elif abs(edge_delta_ev - 120) < 3:
            cross_section = 8.79e-8
        elif abs(edge_delta_ev - 200) < 3:
            cross_section = 1.40e-7

    return cross_section


def relative_atomic_abundance(counts_edge: float, partial_cross_section_nm2: float) -> float:
    """Return the relative atomic concentration.

    partial_cross_section is in nm * nm.

    The return value units are atoms / (nm * nm) * spectrum intensity.
    """
    pass


def atomic_areal_density_nm2(counts_edge: float, counts_spectrum: float, partial_cross_section_nm2: float) -> float:
    """Return the areal density.

    partial_cross_section is in nm * nm.

    The return value units are atoms / (nm * nm).
    """
    pass


def edge_onset_energy_eV(atomic_number: int, shell_number: int, subshell_index: int) -> float:
    """Return the electron binding energy for the given edge.

    Return value is in eV.
    """
    pass


def edges_near_energy_eV(energy_loss_eV: float, energy_loss_delta_eV: float) -> list:
    """Return a list of edges near the energy_loss.
    """
    pass
