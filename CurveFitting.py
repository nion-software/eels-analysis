"""
    Curve Fitting

    A library of functions and classes for general curve fitting techniques.

"""

# standard libraries
import gettext

# third party libraries
import numpy

_ = gettext.gettext


class MultipleLinearRegression1D(object):
    """A tool for performing generic multiple linear regression on one-dimensional data arrays.

    This module follows the Multiple Regression chapter in P.R. Bevington,
    'Data Reduction and Error Analysis for the Physical Sciences'.

    Two different initializers are provided, one for ordered, equispaced data (for which the
    x values are immaterial), and another that allows explicit provision of arbitrarily sampled x values.
    """

    def __init__(self, data, x_values = None):
        """Initialize with measured data and corresponding x values in 1-D arrays.

        If no x values are supplied, then the data are assumed to be equispaced over -1 to 1, inclusive.
        """
        self.data = data.copy()
        if x_values == None:
            self.x_values = numpy.linspace(-1.0, 1.0, num = data.shape[0])
        else:
            self.x_values = x_values.copy()

    def fit_function_set(self, function_set):
        """Fit data with a linear combination of the passed function set, returning both the coefficient and fit data arrays."""
        assert function_set.shape[0] == data.shape[0]
        return self.data

    def fit_polynomial(self, poynomial_order = 1):
        """Fit data with a polynomial of the specified order, returning both the coefficient and fit data arrays."""
        return self.data


class PolynomialFit1D(object):
    """A utility class for performing generic polynomial fitting on one-dimensional data arrays.

    This module follows the chapter "Least-Squares Fit to a Polynomial" in P.R. Bevington,
    'Data Reduction and Error Analysis for the Physical Sciences'.

    The fit must be initialized with a 1D array of y values plus either a corresponding 1D array of x values,
    or (if the y value array is ordered and equispaced) the x value for the first y value and the x increment per sample.

    Exponential, Gaussian, and power-law fits are supported via log-scale flags for the y and x dimensions.
    """

    def __init__(self, y_values: numpy.ndarray, x_values: numpy.ndarray = None, first_x: float = 0, delta_x: float = 1, polynomial_order: int = 1,
                 y_log_scale: bool = False, x_log_scale: bool = False):
        """
            Required: y_values - a 1D NumPy array containing the measured data (ordinate) values.
            Optional: x_values - a 1D NumPy array containing the corresponding independent variable (abscissa) values.
                first_x and delta_x - alternate specification of x (abscissa) values, only applicable when y_values array is ordered and equispaced.
                polynomial_order - non-negative integer value specifying fit polynomial order, e.g. 0 = constant, 1 = line, 2 = parabola, etc.
                y_log_scale - boolean value specifying whether fit should be applied to log of y values, e.g. exponential, Gaussian, or power-law.
                x_log_scale - boolean value specifying whether fit should be applied to log of x values, e.g. logarithmic curve or power-law.
        """

        self.__y_log_scale = y_log_scale
        self.__x_log_scale = x_log_scale

        #   logging.info("Data size: {:d}".format(y_values.shape[0]))

        # Check validity of y_values array and transfer to fit_ordinates array, applying log, if requested.
        # PROBLEM: using assertions for now, but really need a mechanism to provide helpful guidance to the user
        assert len(y_values.shape) == 1
        sample_count = y_values.shape[0]
        assert sample_count >= polynomial_order + 2
        self.__fit_ordinates = y_values.copy()
        if self.__y_log_scale:
            # for log-scale fitting, all values must be positive
            assert self.__fit_ordinates.min() > 0
            self.__fit_ordinates = numpy.log(self.__fit_ordinates)

        # Check validity of x_values array, if applicable, and prepare fit_abscissae array, applying log, if requested.
        if x_values is not None:
            # x value array must match y value array in size
            assert len(x_values.shape) == 1
            assert x_values.shape[0] == sample_count
            self.__fit_abscissae = numpy.copy(x_values)
            self.__fit_abscissa_min = numpy.amin(self.__fit_abscissae)
            self.__fit_abscissa_max = numpy.amax(self.__fit_abscissae)
        else:
            assert delta_x > 0
            self.__fit_abscissa_min = first_x
            self.__fit_abscissa_max = first_x + delta_x * (sample_count - 1)
            self.__fit_abscissae = numpy.linspace(self.__fit_abscissa_min, self.__fit_abscissa_max, sample_count)

        if self.__x_log_scale:
            # for log-scale fitting, all values must be positive
            assert self.__fit_abscissa_min > 0
            self.__fit_abscissa_min = numpy.log(self.__fit_abscissa_min)
            self.__fit_abscissa_max = numpy.log(self.__fit_abscissa_max)
            self.__fit_abscissae = numpy.log(self.__fit_abscissae)

        # To keep computations well-conditioned, it is best to transform the abscissae to the range -1 to 1, inclusive
        self.__fit_abscissa_scale = 2.0 / (self.__fit_abscissa_max - self.__fit_abscissa_min)
        # rewrite this below to avoid PyCharm type checking errors during editing
        # self.__fit_abscissae = self.__fit_abscissa_scale * (self.__fit_abscissae - self.__fit_abscissa_min) - 1
        self.__fit_abscissae = numpy.subtract(numpy.multiply(self.__fit_abscissa_scale, numpy.subtract(self.__fit_abscissae, self.__fit_abscissa_min)), 1)

        self.__set_polynomial_order(polynomial_order)

    def __get_polynomial_order(self):
        """The order of the fit polynomial, e.g. 1 = line, 2 = parabola, 3 = cubic, etc."""
        return self.fit_polynomial.o

    def __set_polynomial_order(self, polynomial_order):
        assert polynomial_order >= 0
        self.fit_polynomial = numpy.poly1d(numpy.polyfit(self.__fit_abscissae, self.__fit_ordinates, polynomial_order))

    polynomial_order = property(__get_polynomial_order, __set_polynomial_order)

    def compute_fit_for_values(self, values):
        if self.__x_log_scale:
            assert values.min() > 0
            abscissae = numpy.log(values)
        else:
            abscissae = values

        abscissae = self.__fit_abscissa_scale * (abscissae - self.__fit_abscissa_min) - 1
        computed_fit = self.fit_polynomial(abscissae)
        if self.__y_log_scale:
            computed_fit = numpy.exp(computed_fit)
        return computed_fit

    def compute_fit_for_range(self, fit_range, point_count):
        assert fit_range.min() > 0
        values = numpy.linspace(fit_range[0], fit_range[1], point_count)
        return self.compute_fit_for_values(values)
