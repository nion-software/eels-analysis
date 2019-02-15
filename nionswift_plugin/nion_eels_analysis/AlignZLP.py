# imports
import numpy
import typing
import scipy.ndimage

# local libraries
from nion.data import DataAndMetadata


def align_zlp_xdata(src_xdata: DataAndMetadata.DataAndMetadata, progress_fn=None) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
    # check to make sure it is suitable for this algorithm
    if src_xdata.is_datum_1d and (src_xdata.is_sequence or src_xdata.is_collection):

        # get the numpy array and create the destination data
        src_data = src_xdata.data
        dst_data = numpy.zeros_like(src_data)

        # set up the indexing. to make this algorithm work with any indexing,
        # we will iterate over all non-datum dimensions using numpy.unravel_index.
        d_rank = src_xdata.datum_dimension_count
        src_shape = tuple(src_xdata.data_shape)
        s_shape = src_shape[0:-d_rank]
        count = int(numpy.product(s_shape))

        # use this as the reference position. all other spectra will be aligned to this one.
        ref_pos = numpy.argmax(src_data[(0,)*len(s_shape)])

        # loop over all non-datum dimensions linearly
        for i in range(count):
            # generate the index for the non-datum dimensions using unravel_index
            ii = numpy.unravel_index(i, s_shape)

            # the algorithm in this early version is to find the max value
            mx_pos = numpy.argmax(src_data[ii])

            # determine the offset (an integer) and store the shifted data into the result
            offset = mx_pos - ref_pos
            if offset < 0:
                dst_data[ii][-offset:] = src_data[ii][0:offset]
            elif offset > 0:
                dst_data[ii][:-offset] = src_data[ii][offset:]
            else:
                dst_data[ii][:] = src_data[ii][:]

            # if the last index is 0, report progress
            if ii[-1] == 0 and ii[0] % 10 == 0:
                if callable(progress_fn):
                    progress_fn(ii[0])

        dimensional_calibrations = src_xdata.dimensional_calibrations
        energy_calibration = dimensional_calibrations[-1]
        energy_calibration.offset = -(ref_pos + 0.5) * energy_calibration.scale
        dimensional_calibrations = dimensional_calibrations[0:-1] + [energy_calibration]

        # dst_data is complete. construct xdata with correct calibration and data descriptor.
        data_descriptor = DataAndMetadata.DataDescriptor(src_xdata.is_sequence, src_xdata.collection_dimension_count, 1)
        return DataAndMetadata.new_data_and_metadata(dst_data, src_xdata.intensity_calibration, dimensional_calibrations, data_descriptor=data_descriptor)

    return None

def gaussian(x, a, b, c):
    return a*numpy.e**(-(x-b)**2/(2*c**2))


def jac_gaussian(x, a, b, c):
    exp = numpy.e**(-(-b + x)**2/(2*c**2))
    return numpy.swapaxes(numpy.array((exp, -a*(2*b - 2*x)*exp/(2*c**2), a*(-b + x)**2*exp/c**3)), 0, 1)


def estimate_zlp_amplitude_position_width_fit_spline(d):
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
        popt, pcov = scipy.optimize.curve_fit(gaussian, numpy.arange(r[0], r[1]), d[r[0]:r[1]], p0=p0,
                                              jac=jac_gaussian)#, bounds=([d_max * 0.9, r[0], c * 0.9], [d_max * 1.1, r[1], c * 1.1]))
        return popt
    return numpy.nan, numpy.nan, numpy.nan

def estimate_zlp_amplitude_position_width_com(d):

    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    mx_pos = numpy.argmax(d)
    mx = d[mx_pos]
    half_mx = mx/2
    left_pos = mx_pos - numpy.sum(d[:mx_pos] > half_mx)
    right_pos = mx_pos + numpy.sum(d[mx_pos:] > half_mx)
    mx_pos_sub = numpy.sum(d[left_pos:right_pos] * numpy.arange(right_pos - left_pos))/numpy.sum(d[left_pos:right_pos])
    return mx, mx_pos_sub + left_pos, left_pos, right_pos

def align_zlp_xdata_subpixel(src_xdata: DataAndMetadata.DataAndMetadata, progress_fn=None, method='com') -> typing.Optional[DataAndMetadata.DataAndMetadata]:
    # check to make sure it is suitable for this algorithm
    if src_xdata.is_datum_1d and (src_xdata.is_sequence or src_xdata.is_collection):

        # get the numpy array and create the destination data
        src_data = src_xdata.data

        # set up the indexing. to make this algorithm work with any indexing,
        # we will iterate over all non-datum dimensions using numpy.unravel_index.
        d_rank = src_xdata.datum_dimension_count
        src_shape = tuple(src_xdata.data_shape)
        d_shape = src_shape[-d_rank:]

        flat_src_data = numpy.reshape(src_data, (-1,) + d_shape)
        flat_dst_data = numpy.zeros_like(flat_src_data)

        get_position_fun = (estimate_zlp_amplitude_position_width_com if method == 'com'
                            else estimate_zlp_amplitude_position_width_fit_spline)
        # use this as the reference position. all other spectra will be aligned to this one.
        ref_pos = get_position_fun(flat_src_data[0])[1]
        # put the first spectrum in the result
        flat_dst_data[0] = flat_src_data[0]
        # loop over all non-datum dimensions linearly
        for i in range(1, len(flat_src_data)):
            # the algorithm in this early version is to find the max value
            #mx_pos = MeasureZLP.estimate_zlp_amplitude_position_width_fit_spline(flat_src_data[i])[1]
            mx_pos = get_position_fun(flat_src_data[i])[1]
            # fallback to simple max if get_position_fun failed
            if mx_pos is numpy.nan:
                mx_pos = numpy.argmax(flat_src_data[i])
            # determine the offset and apply it
            offset = ref_pos - mx_pos
            src_data_fft = numpy.fft.fftn(flat_src_data[i])
            flat_dst_data[i] = numpy.fft.ifftn(scipy.ndimage.fourier_shift(src_data_fft, offset)).real
            # every row, report progress (will also work for a sequence or 1d collection
            # because there we have only 1 row anyways)
            if i % src_shape[1] == 0 and callable(progress_fn):
                progress_fn(i//src_shape[1])

        dimensional_calibrations = src_xdata.dimensional_calibrations
        energy_calibration = dimensional_calibrations[-1]
        energy_calibration.offset = -(ref_pos + 0.5) * energy_calibration.scale
        dimensional_calibrations = dimensional_calibrations[:-1] + [energy_calibration]

        # dst_data is complete. construct xdata with correct calibration and data descriptor.
        data_descriptor = DataAndMetadata.DataDescriptor(src_xdata.is_sequence, src_xdata.collection_dimension_count, 1)
        return DataAndMetadata.new_data_and_metadata(flat_dst_data.reshape(src_shape), src_xdata.intensity_calibration, dimensional_calibrations, data_descriptor=data_descriptor)

    return None


def align_zlp(api, window):
    # find the focused data item
    src_data_item = window.target_data_item
    if src_data_item:

        def progress(i):
            print(f"Processing row {i} (align zlp)")

        dst_xdata = align_zlp_xdata(src_data_item.xdata, progress)

        if dst_xdata:
            # create a new data item in the library and set its title.
            data_item = api.library.create_data_item_from_data_and_metadata(dst_xdata)
            data_item.title = "Aligned (max) " + src_data_item.title

            # display the data item.
            window.display_data_item(data_item)
        else:
            print("Failed: Data is not a sequence or collection of 1D spectra.")
    else:
        print("Failed: No data item selected.")


def align_zlp_com(api, window):
    # find the focused data item
    src_data_item = window.target_data_item
    if src_data_item:

        def progress(i):
            print(f"Processing row {i} (align zlp)")

        dst_xdata = align_zlp_xdata_subpixel(src_data_item.xdata, progress, method='com')

        if dst_xdata:
            # create a new data item in the library and set its title.
            data_item = api.library.create_data_item_from_data_and_metadata(dst_xdata)
            data_item.title = "Aligned (com) " + src_data_item.title

            # display the data item.
            window.display_data_item(data_item)
        else:
            print("Failed: Data is not a sequence or collection of 1D spectra.")
    else:
        print("Failed: No data item selected.")


def align_zlp_fit(api, window):
    # find the focused data item
    src_data_item = window.target_data_item
    if src_data_item:

        def progress(i):
            print(f"Processing row {i} (align zlp)")

        dst_xdata = align_zlp_xdata_subpixel(src_data_item.xdata, progress, method='fit')

        if dst_xdata:
            # create a new data item in the library and set its title.
            data_item = api.library.create_data_item_from_data_and_metadata(dst_xdata)
            data_item.title = "Aligned (peak fit) " + src_data_item.title

            # display the data item.
            window.display_data_item(data_item)
        else:
            print("Failed: Data is not a sequence or collection of 1D spectra.")
    else:
        print("Failed: No data item selected.")
