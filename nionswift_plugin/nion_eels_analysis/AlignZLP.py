# system imports
import gettext

# third part imports
import numpy

# local libraries
from nion.data import DataAndMetadata

_ = gettext.gettext



def align_zlp(api, window):
    # find the focused data item
    src_data_item = window.target_data_item
    if src_data_item:

        src_xdata = src_data_item.xdata
        # check to make sure it is suitable for this algorithm
        if src_xdata.is_datum_1d and (src_xdata.is_sequence or src_xdata.is_collection):

            # get the numpy array and create the destination data
            src_data = src_data_item.data
            dst_data = numpy.zeros(src_data.shape)

            # set up the indexing. to make this algorithm work with any indexing,
            # we will iterate over all non-datum dimensions using numpy.unravel_index.
            d_rank = src_xdata.datum_dimension_count
            src_shape = tuple(src_xdata.data_shape)
            s_shape = src_shape[0:-d_rank]
            count = int(numpy.product(s_shape))

            # use this as the reference position. all other spectra will be aligned to this one.
            ref_pos = numpy.argmax(src_data[0, 0])

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
                    print(f"Processing row {ii[0]} (align zlp)")

            dimensional_calibrations = src_xdata.dimensional_calibrations
            energy_calibration = dimensional_calibrations[-1]
            energy_calibration.offset = -(ref_pos + 0.5) * energy_calibration.scale
            dimensional_calibrations = dimensional_calibrations[0:-1] + [energy_calibration]

            # dst_data is complete. construct xdata with correct calibration and data descriptor.
            dst_xdata = DataAndMetadata.new_data_and_metadata(dst_data, src_xdata.intensity_calibration, dimensional_calibrations, data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 1))

            # create a new data item in the library and set its title.
            data_item = api.library.create_data_item_from_data_and_metadata(dst_xdata)
            data_item.title = "Aligned " + src_data_item.title

            # display the data item.
            window.display_data_item(data_item)
        else:
            print("Failed: Data is not a sequence or collection of 1D spectra.")
    else:
        print("Failed: No data item selected.")


class MenuItemDelegate:

    def __init__(self, api):
        self.__api = api
        self.menu_id = "eels_menu"  # required, specify menu_id where this item will go
        self.menu_name = _("EELS")  # optional, specify default name if not a standard menu
        self.menu_before_id = "window_menu"  # optional, specify before menu_id if not a standard menu
        self.menu_item_name = _("Align ZLP (max method)")  # menu item name

    def menu_item_execute(self, window):
        align_zlp(self.__api, window)


class MenuExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.eels_analysis.menu_item_align_zlp"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="~1.0")
        # be sure to keep a reference or it will be closed immediately.
        self.__menu_item_ref = api.create_menu_item(MenuItemDelegate(api))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__menu_item_ref.close()
        self.__menu_item_ref = None
