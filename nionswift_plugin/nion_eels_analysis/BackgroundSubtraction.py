# system imports
import gettext

# third part imports
import numpy

# local libraries
from nion.data import xdata_1_0 as xd
from nion.swift.model import Symbolic
from nion.eels_analysis import eels_analysis

_ = gettext.gettext


class EELSBackgroundSubtraction:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, eels_spectrum_data_item, fit_interval_graphics, signal_interval_graphic):
        eels_spectrum_xdata = eels_spectrum_data_item.xdata
        fit_intervals = [fit_interval_graphic.interval for fit_interval_graphic in fit_interval_graphics]
        signal_interval = signal_interval_graphic.interval
        signal_xdata = eels_analysis.extract_original_signal(eels_spectrum_xdata, fit_intervals, signal_interval)
        background_xdata = eels_analysis.calculate_background_signal(eels_spectrum_xdata, fit_intervals, signal_interval)
        subtracted_xdata = signal_xdata - background_xdata
        offset = int(round((signal_interval[0] - fit_intervals[0][0]) * eels_spectrum_xdata.data_shape[0]))
        length = int(round((signal_interval[1] - signal_interval[0]) * eels_spectrum_xdata.data_shape[0]))
        background_xdata = eels_analysis.make_signal_like(background_xdata, eels_spectrum_xdata)
        subtracted_xdata = eels_analysis.make_signal_like(subtracted_xdata[offset:offset + length], eels_spectrum_xdata)
        self.__xdata = xd.vstack((eels_spectrum_xdata, background_xdata, subtracted_xdata))

    def commit(self):
        self.computation.set_referenced_xdata("data", self.__xdata)


class EELSMapping:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, spectrum_image_data_item, fit_interval_graphics, signal_interval_graphic):
        spectrum_image_xdata = spectrum_image_data_item.xdata
        fit_intervals = [fit_interval_graphic.interval for fit_interval_graphic in fit_interval_graphics]
        signal_interval = signal_interval_graphic.interval
        self.__mapped_xdata = eels_analysis.map_background_subtracted_signal(spectrum_image_xdata, None, fit_intervals, signal_interval)

    def commit(self):
        self.computation.set_referenced_xdata("map", self.__mapped_xdata)


async def use_interval_as_signal(api, window):
    target_data_item = window.target_data_item
    target_display = window.target_display
    target_graphic = target_display.selected_graphics[0] if target_display and len(target_display.selected_graphics) == 1 else None
    target_interval = target_graphic if target_graphic and target_graphic.graphic_type == "interval-graphic" else None
    if target_data_item and target_interval:
        interval = target_interval.interval
        output = api.library.create_data_item(title="{} Combined".format(target_data_item.title))
        target_interval = output.add_interval_region(interval[0], interval[1])
        fit_ahead = output.add_interval_region(interval[0] * 0.8, interval[0] * 0.9)
        fit_behind = output.add_interval_region(interval[1] * 1.1, interval[1] * 1.2)
        computation = api.library.create_computation("eels.background_subtraction3", inputs={"eels_spectrum_data_item": target_data_item, "fit_interval_graphics": [fit_ahead, fit_behind], "signal_interval_graphic": target_interval}, outputs={"data": output})
        computation._computation.source = target_interval._graphic
        target_interval._graphic.source = computation._computation
        fit_ahead._graphic.source = target_interval._graphic
        fit_behind._graphic.source = target_interval._graphic
        fit_ahead.label = "background"
        fit_behind.label = "background"
        target_interval.label = "signal"
        target_interval._graphic.color = "#0F0"
        output.display._display.legend_labels = ["Data", "Background", "Signal"]
        window.display_data_item(output)
        # the composite item will need the initial computation results to display properly (view to intervals)
        from nion.swift.model import DataItem
        document_controller = window._document_controller
        document_model = document_controller.document_model
        await document_model.compute_immediate(document_controller.event_loop, computation._computation)
        display_specifier = DataItem.DisplaySpecifier.from_data_item(output._data_item)
        display_specifier.display.view_to_intervals(output._data_item.xdata, [fit_ahead._graphic.interval, fit_behind._graphic.interval])


def use_signal_for_map(api, window):
    target_data_item = window.target_data_item
    target_display = window.target_display
    target_graphic = target_display.selected_graphics[0] if target_display and len(target_display.selected_graphics) == 1 else None
    target_interval = target_graphic if target_graphic and target_graphic.graphic_type == "interval-graphic" else None
    if target_data_item and target_interval:
        for computation in api.library._document_model.computations:
            if computation.processing_id == "eels.background_subtraction3" and target_interval._graphic in computation._inputs:
                fit_interval_graphics = computation._get_variable("fit_interval_graphics").bound_item.value
                signal_interval_graphic = computation._get_variable("signal_interval_graphic").bound_item.value
                spectrum_image = api._new_api_object(api.library._document_model.get_source_data_items(api.library._document_model.get_source_data_items(target_data_item._data_item)[0])[0])
                map = api.library.create_data_item_from_data(numpy.zeros_like(spectrum_image.display_xdata.data), title="{} Map".format(spectrum_image.title))
                fit_interval_graphics = [api._new_api_object(g) for g in fit_interval_graphics]
                signal_interval_graphic = api._new_api_object(signal_interval_graphic)
                computation = api.library.create_computation("eels.mapping2", inputs={"spectrum_image_data_item": spectrum_image, "fit_interval_graphics": fit_interval_graphics, "signal_interval_graphic": signal_interval_graphic}, outputs={"map": map})
                computation._computation.source = target_interval._graphic
                window.display_data_item(map)


class MenuItemDelegate:

    def __init__(self, api):
        self.__api = api
        self.menu_id = "eels_menu"  # required, specify menu_id where this item will go
        self.menu_name = _("EELS")  # optional, specify default name if not a standard menu
        self.menu_before_id = "window_menu"  # optional, specify before menu_id if not a standard menu
        self.menu_item_name = _("Subtract Background from Signal")  # menu item name

    def menu_item_execute(self, window):
        window._document_controller.event_loop.create_task(use_interval_as_signal(self.__api, window))

class MenuItemDelegate2:

    def __init__(self, api):
        self.__api = api
        self.menu_id = "eels_menu"  # required, specify menu_id where this item will go
        self.menu_name = _("EELS")  # optional, specify default name if not a standard menu
        self.menu_before_id = "window_menu"  # optional, specify before menu_id if not a standard menu
        self.menu_item_name = _("Map Signal")  # menu item name

    def menu_item_execute(self, window):
        use_signal_for_map(self.__api, window)


class MenuExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.eels_analysis.menu_item_background"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="~1.0")
        # be sure to keep a reference or it will be closed immediately.
        self.__menu_item_ref = api.create_menu_item(MenuItemDelegate(api))
        self.__menu_item_ref2 = api.create_menu_item(MenuItemDelegate2(api))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__menu_item_ref.close()
        self.__menu_item_ref = None
        self.__menu_item_ref2.close()
        self.__menu_item_ref2 = None


Symbolic.register_computation_type("eels.background_subtraction3", EELSBackgroundSubtraction)
Symbolic.register_computation_type("eels.mapping2", EELSMapping)
