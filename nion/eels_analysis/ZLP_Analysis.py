"""
A library of functions for finding and characterizing the zero-loss peak
"""
import logging
import numpy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import scipy.interpolate


def gaussian(x, a, b, c):
    return a*numpy.e**(-(x-b)**2/(2*c**2))


def jac_gaussian(x, a, b, c):
    exp = numpy.e**(-(-b + x)**2/(2*c**2))
    return numpy.swapaxes(numpy.array((exp, -a*(2*b - 2*x)*exp/(2*c**2), a*(-b + x)**2*exp/c**3)), 0, 1)


def estimate_zlp_amplitude_position_width_fit_spline(d):
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
            return popt
    return numpy.nan, numpy.nan, numpy.nan


def estimate_zlp_amplitude_position_width_com(d):

    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    assert len(d.shape) == 1
    mx_pos = numpy.argmax(d)
    mx = d[mx_pos]
    half_mx = mx/2
    left_pos = mx_pos - numpy.sum(d[:mx_pos] > half_mx)
    right_pos = mx_pos + numpy.sum(d[mx_pos:] > half_mx)
    mx_pos_sub = numpy.sum(d[left_pos:right_pos] * numpy.arange(right_pos - left_pos))/numpy.sum(d[left_pos:right_pos])
    return mx, mx_pos_sub + left_pos, left_pos, right_pos