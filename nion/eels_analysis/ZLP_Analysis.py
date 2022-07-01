"""
A library of functions for finding and characterizing the zero-loss peak
"""
import logging
import math
import numpy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import scipy.interpolate
import typing


DataArrayType = numpy.typing.NDArray[typing.Any]


def gaussian(x: DataArrayType, a: DataArrayType, b: DataArrayType, c: DataArrayType) -> DataArrayType:
    return typing.cast(DataArrayType, a * numpy.e ** (-(x - b) ** 2 / (2 * c ** 2)))


def jac_gaussian(x: DataArrayType, a: DataArrayType, b: DataArrayType, c: DataArrayType) -> DataArrayType:
    exp = math.e ** (-(-b + x) ** 2 / (2 * c ** 2))
    return numpy.swapaxes(numpy.array((exp, -a * (2 * b - 2 * x) * exp / (2 * c ** 2), a * (-b + x) ** 2 * exp / c ** 3)), 0, 1)


def estimate_zlp_amplitude_position_width_fit_spline(d: DataArrayType) -> typing.Tuple[float, float, float]:
    assert len(d.shape) == 1
    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    #gaussian = lambda x, a, b, c: a*numpy.exp(-(x-b)**2/(2*c**2))
    max_pos = numpy.argmax(d)
    d_max = d[max_pos]
    # first fit a bspline to the data
    s = scipy.interpolate.splrep(range(d.shape[-1]), d - d_max / 2)
    # assuming bspline has two roots, use them to estimate FWHM
    r = scipy.interpolate.sproot(s).astype(int)
    if len(r) == 2:
        fwhm = r[1] - r[0]
        p0 = (d_max, max_pos, fwhm/2)
        #c = fwhm / (2 * math.sqrt(2 * math.log(2)))
        # now fit the gaussian to the data, using the amplitude, std dev, and bspline position as estimates (10%)
        try:
            popt, pcov = scipy.optimize.curve_fit(gaussian, numpy.arange(r[0], r[1]), d[r[0]:r[1]], p0=p0,
                                                  jac=jac_gaussian)
        except RuntimeError as e:
            logging.error(str(e))
        else:
            return typing.cast(typing.Tuple[float, float, float], popt)
    return numpy.nan, numpy.nan, numpy.nan


def estimate_zlp_amplitude_position_width_com(d: DataArrayType) -> typing.Tuple[float, float, float, float]:
    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    assert len(d.shape) == 1
    mx_pos = typing.cast(int, numpy.argmax(d))
    mx = d[mx_pos]
    quarter_max = mx * 0.25
    left_pos = mx_pos - numpy.sum(d[:mx_pos] > quarter_max) * 3
    right_pos = mx_pos + numpy.sum(d[mx_pos:] > quarter_max) * 3
    left_pos = max(0, left_pos)
    right_pos = min(d.size, right_pos)
    d_sub = d[left_pos:right_pos] - quarter_max
    d_sub[d_sub < 0] = 0
    mx_pos_sub = numpy.sum(d_sub * numpy.arange(right_pos - left_pos))/(numpy.sum(d_sub) or 1)
    # Also calculate FWHM because it is used by some users of this function
    half_max = mx * 0.5
    left_end = mx_pos - numpy.sum(d[:mx_pos] > half_max)
    right_end = mx_pos + numpy.sum(d[mx_pos:] > half_max)
    return mx, mx_pos_sub + left_pos, left_end, right_end
