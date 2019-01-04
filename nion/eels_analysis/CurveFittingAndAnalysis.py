"""
    Curve Fitting and Analysis

    A library of classes and functions for general curve fitting and analysis techniques applied to arrays of spectra.
"""

import numpy

class MultipleCurveFit:
    """A class for performing multiple linear regression on arrays of 1D data (i.e. curves or spectra).

    The fit must be initialized with a 1D array of model curves (2 dimensions, total), where the model data extends along the last dimension.
    This model matrix is immediately transformed via singular value decomposition (SVD) into a robust fit matrix that can be applied to any
    corresponding collection of curves (e.g. spectra arranged in a 0, 1, or 2-dimensional array, with count data along the last dimension,
    sampled at exactly the same x-axis values as the model curves), thereby yielding coefficients for a least squares fit.

    In order to ensure a well-conditioned fit matrix, each model curve is normalized so that its RMS value is 1.  The inverse scaling factors
    are applied to any fit-coefficient vector returned for external use (see below).

    The fit for a given input data set is generated via the compute_fit_for_data function, which updates the normalized fit coefficients array.
    The fit object can then be queried for specific fit results via the following methods:
        get_fit_coefficients - fit coefficient array (with respect to original, non-normalized model curves)
        get_fit_integrals - fit integral array (e.g. edge count line profiles and maps)
        get_fit_curves - fit curve array (for direct comparison with input data curves)
    """

    def __init__(self, model_curves: numpy.ndarray):
        """
            model_curves - 1D array of 1D model curves (i.e. 2D NumPy ndarray), where the model data extends along the last dimension.
        """
        assert model_curves.ndim == 2
        self.sample_count = model_curves.shape[-1]

        # Determine the RMS value of each model curve and normalize by this factor to help ensure a well-conditioned solution matrix
        # Note that a model curve with RMS value 0 is meaningless and results in a failed assertion
        self._model_rms_values = numpy.sqrt(numpy.sum(numpy.square(model_curves), -1, keepdims = True) / self.sample_count)
        assert numpy.amin(self._model_rms_values) > 0
        self._normalized_model_curves = model_curves / self._model_rms_values
        self._normalized_model_integrals = self._normalized_model_curves.sum(-1, keepdims = True)

        # Generate singular value decomposition of normalized model curve matrix
        u, s, v = numpy.linalg.svd(self._normalized_model_curves.T, full_matrices = False)

        # Invert singular values, zeroing out any that are smaller than 1e-6 times the largest to yield a well-conditioned fit matrix
        s_min = 1e-6 * numpy.amax(s)
        s_inv = numpy.zeros_like(s)
        s_inv[s > s_min] = 1 / s[s > s_min]

        # Generate normalized fit matrix
        self._normalized_fit_matrix = numpy.dot(v, numpy.dot(numpy.diag(s_inv), u.T))

        # Track whether a fit has been computed for a specific data set
        self._have_computed_fit_for_data = False

    def compute_fit_for_data(self, data_values: numpy.ndarray) -> numpy.ndarray:
        """
            data_values - array of 1D data curves (single spectrum, line scan, or area scan), at x values matching those of the model curves.
        """
        assert data_values.shape[-1] == self.sample_count
        self._normalized_fit_coefficients = numpy.einsum('ij, ...j', self._normalized_fit_matrix, data_values)
        self._have_computed_fit_for_data = True

    def get_fit_coefficients(self) -> numpy.ndarray:
        assert self._have_computed_fit_for_data
        fit_coefficients = self._normalized_fit_coefficients / self._model_rms_values.T
        return fit_coefficients

    def get_fit_integrals(self) -> numpy.ndarray:
        assert self._have_computed_fit_for_data
        fit_integrals = numpy.einsum('ij, ...i', self._normalized_model_integrals, self._normalized_fit_coefficients)
        return fit_integrals

    def get_fit_curves(self) -> numpy.ndarray:
        assert self._have_computed_fit_for_data
        fit_curves = numpy.einsum('ij, ...i', self._normalized_model_curves, self._normalized_fit_coefficients)
        return fit_curves


class PolynomialCurveFit:
    """A class for fitting 1D polynomials to arrays of 1D data (i.e. curves or spectra).

    This class uses the MultipleCurveFit class to fit 1D polynomials of specified order to arbitrarily sampled data curves.
    The resulting fit coefficients allow fitted polynomials to be returned for any set of x values,
    as needed for interpolation or extrapolation of backgrounds for signal extraction applications.

    The fit must be initialized with a 1D array of x values corresponding to the sampling of data arrays passed to compute_fit_for_data.
    Linear fitting is done by default, but higher order polynomial fitting can be specified via the polynomial_order parameter.
    Polynomial fits to curves plotted against a logarithmic x axis (e.g. power laws) are specified by setting fit_log_x to True.
    In this case, the x values must all be strictly greater than 0.

    For a given set of x values, multiple data sets can be fitted with the same fit object via the compute_fit_for_data method.
    Data sets that follow a polynomial model when plotted against a logarithmic y (intensity) axis, e.g. exponential, Gaussian,
    and power-law models, can be fitted by passing True for the fit_log_data parameter of this method.
    In this case, all the data values must be strictly greater than 0.
    """

    def _compute_polynomial_model(self, x_values: numpy.ndarray) -> numpy.ndarray:
        """Prepare polynomial model array for the current polynomial order and log-scale settings.
        """
        polynomial_model = numpy.ones([self._polynomial_order + 1, x_values.size], x_values.dtype)
        if self._polynomial_order > 0:
            if self._fit_log_x:
                # For log x fit, all x values must be positive
                assert numpy.amin(x_values) > 0
                polynomial_model[1, :] = numpy.log(x_values)
            else:
                polynomial_model[1, :] = x_values

            power = 2
            while power <= self._polynomial_order:
                polynomial_model[power:self._polynomial_order, :] *= polynomial_model[1, :]
                power += 1

        return polynomial_model

    def __init__(self, x_values: numpy.ndarray, polynomial_order: int = 1, fit_log_x: bool = False):
        """
            x_values - a 1D NumPy array containing the x coordinates corresponding to each entry in the data arrays passed to compute_fit_for_data.
            polynomial_order - non-negative integer value specifying fit polynomial order, e.g. 0 = constant, 1 = line, 2 = parabola, etc.
            fit_log_x - boolean value specifying whether fit should be versus log of x values, e.g. logarithmic curve or power-law.
        """
        assert x_values.ndim == 1
        assert x_values.var() > 0
        assert x_values.size >= polynomial_order + 1

        self._polynomial_order = polynomial_order
        self._fit_log_x = fit_log_x

        self._multicurve_fit = MultipleCurveFit(self._compute_polynomial_model(x_values))

    def compute_fit_for_data(self, data_values: numpy.ndarray, fit_log_data: bool = False) -> numpy.ndarray:
        """
            data_values - array of 1D data curves (single spectrum, line scan, or area scan), at x values specified at fit initialization.
            fit_log_data - boolean value specifying whether fit should be to log of y values, e.g. exponential, Gaussian, or power-law.
        """
        assert data_values.shape[-1] == self._multicurve_fit.sample_count

        self._fit_log_y = fit_log_data

        y_values = data_values.copy()
        if self._fit_log_y:
            # For log y fit, all values must be positive
            assert numpy.amin(y_values) > 0
            y_values = numpy.log(y_values)

        self._multicurve_fit.compute_fit_for_data(y_values)

    def get_fit_coefficients(self) -> numpy.ndarray:
        return self._multicurve_fit.get_fit_coefficients()

    def evaluate_fit_at(self, x_values) -> numpy.ndarray:
        evaluated_fit = numpy.einsum('ij, ...i', self._compute_polynomial_model(x_values), self._multicurve_fit.get_fit_coefficients())
        if self._fit_log_y:
            evaluated_fit = numpy.exp(evaluated_fit)
        return evaluated_fit


class RangeSliceConverter:
    """A class for converting between calibrated ranges and slices on equispaced 1D data arrays.

    A range is specified as a 2-element array of the form [range_start, range_end], where range_start is the
    coordinate of the first data element in the range and range_end is that just after last data element in the range.
    This means that the number of data samples in a range is given by (range_end - range_start) / coordinate_step,
    where coordinate_step is the coordinate increment between successive data samples.  Another way to think about
    this is that range_start is the left edge of the first data channel, while range_end is the right edge of the last.

    Each converter must be initialized with the coordinate value at array index 0 and the coordinate step per array index increment.
    """

    def __init__(self, coordinate_at_0: float, coordinate_step: float):
        assert coordinate_step > 0
        self._origin = coordinate_at_0
        self._step = coordinate_step

    def get_slice(self, range_for_slice: numpy.ndarray) -> slice:
        assert range_for_slice.ndim == 1
        assert range_for_slice.size == 2
        slice_start = round((range_for_slice[0] - self._origin) / self._step)
        slice_stop = round((range_for_slice[1] - self._origin) / self._step)
        return slice(int(slice_start), int(slice_stop))

    def get_range(self, slice_for_range: slice) -> numpy.ndarray:
        range_start = self._origin + slice_for_range.start * self._step
        range_end = self._origin + slice_for_range.stop * self._step
        return numpy.array([range_start, range_end])


def signal_from_polynomial_background(data_values: numpy.ndarray, data_x_range: numpy.ndarray, signal_x_range: numpy.ndarray,
                                                background_fit_x_ranges: numpy.ndarray, polynomial_order: int = 1,
                                                fit_log_data: bool = False, fit_log_x: bool = False) -> tuple:
    """Extracts signal from polynomial background fitted to an array of uniformly sampled 1D data curves, returning both the signal and background arrays.

    Primary inputs:
        data_values - a (possibly multi-dimensional) array of uniformly sampled 1D data sets, with samples arranged along the last array dimension
        data_x_range - range of equispaced x coordinates at which all of the 1D data sets are sampled
        signal_x_range - range of x coordinates over which the signal of interest occurs
        background_fit_x_ranges - one or more (possibly overlapping or non-contiguous) ranges that define the signal background to be modelled

    All range parameters are given as 2-element arrays of the form [x_start, x_end], where x_start is the x coordinate
    of the first data element in the range and x_end is the x coordinate just after last data element in the range.
    Multiple background ranges are given as successive rows in a 2D array.

    A fitted polynomial background is evaluated over the contiguous union of the signal and background fit ranges, thereby
    yielding the background under the signal by either extrapolation or interpolation, depending on the range relationships.
    This evaluated background is subtracted from the data curves to yield the net signal in each.

    Optional inputs:
        polynomial_order - order of the polynomial model function (i.e. 0: constant, 1: line (default), 2: parabola, etc).
        fit_log_data - pass True to fit a polynomial to the log of the data values (e.g. exponential, Gaussian tail, or power-law fit)
        fit_log_x - pass True to perform the fit with respect to the log of the x values (e.g. logarithmic or power-law fit)

    Returns:
        signal_integral - net signal integral array after subtraction of background fit over the specified signal range
        signal_profile - net signal profile array after subtraction of background fit over the profile range (see below)
        background_model - background fit profile array over the profile range (see below)
        profile_range - contiguous union of signal and background fit ranges
    """
    assert data_x_range.ndim == 1
    assert data_x_range.size == 2
    assert data_x_range[0] < data_x_range[1]

    assert signal_x_range.ndim == 1
    assert signal_x_range.size == 2
    assert signal_x_range[0] <= signal_x_range[1]
    assert signal_x_range[0] >= data_x_range[0]
    assert signal_x_range[1] <= data_x_range[1]

    assert background_fit_x_ranges.ndim <= 2
    assert background_fit_x_ranges.shape[-1] == 2
    assert numpy.all(background_fit_x_ranges[..., 0] <= background_fit_x_ranges[..., 1])
    assert background_fit_x_ranges.min() >= data_x_range[0]
    assert background_fit_x_ranges.max() <= data_x_range[1]

    # Distill the fit ranges so that they are ordered, consolidated, and non-overlapping
    sorted_fit_ranges = numpy.atleast_2d(background_fit_x_ranges)
    fit_range_order = numpy.argsort(sorted_fit_ranges, 0)[:, 0]
    sorted_fit_ranges = sorted_fit_ranges[fit_range_order]
    clean_fit_ranges = numpy.atleast_2d(sorted_fit_ranges[0])
    for range_index in range(1, sorted_fit_ranges.shape[0]):
        if clean_fit_ranges[-1, 1] > sorted_fit_ranges[range_index, 0]:
            clean_fit_ranges[-1, 1] = max(clean_fit_ranges[-1, 1], sorted_fit_ranges[range_index, 1])
        else:
            clean_fit_ranges = numpy.append(clean_fit_ranges, sorted_fit_ranges[range_index])
    if clean_fit_ranges.shape[0] > 2:
        clean_fit_ranges = clean_fit_ranges.reshape([clean_fit_ranges.shape[0] // 2, 2])

    # Compile data and x-value arrays over fit ranges for input to the polynomial background fit
    x_origin = data_x_range[0]
    x_step = (data_x_range[1] - x_origin) / data_values.shape[-1]
    data_range_converter = RangeSliceConverter(x_origin, x_step)
    x_values = numpy.arange(x_origin, data_x_range[1], x_step, dtype=numpy.float32)
    next_slice = data_range_converter.get_slice(clean_fit_ranges[0])
    x_values_for_fit = x_values[next_slice]
    data_values_for_fit = numpy.maximum(data_values[..., next_slice], 1)
    for range_index in range(1, clean_fit_ranges.shape[0]):
        next_slice = data_range_converter.get_slice(clean_fit_ranges[range_index])
        x_values_for_fit = numpy.append(x_values_for_fit, x_values[next_slice])
        data_values_for_fit = numpy.append(data_values_for_fit, numpy.maximum(data_values[..., next_slice], 1), axis=-1)

    # Generate the requested polynomial fit for the specified fit ranges
    background_fit = PolynomialCurveFit(x_values_for_fit, polynomial_order, fit_log_x)
    background_fit.compute_fit_for_data(data_values_for_fit, fit_log_data)

    # Establish the net profile range, i.e. the contiguous union of fit and signal ranges
    profile_range = numpy.zeros_like(signal_x_range)
    profile_range[0] = min(signal_x_range[0], clean_fit_ranges.min())
    profile_range[1] = max(signal_x_range[1], clean_fit_ranges.max())
    profile_slice = data_range_converter.get_slice(profile_range)

    # Evaluate background model over the net profile range
    background_model = background_fit.evaluate_fit_at(x_values[profile_slice])

    # Compute the net signal profile
    signal_profile = data_values[..., profile_slice] - background_model

    # Compute the net signal integral over the specified signal range
    profile_range_converter = RangeSliceConverter(profile_range[0], x_step)
    signal_slice = profile_range_converter.get_slice(signal_x_range)
    signal_integral = numpy.trapz(signal_profile[..., signal_slice], dx = x_step)

    return signal_integral, signal_profile, background_model, profile_range

