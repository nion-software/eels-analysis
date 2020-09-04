from __future__ import annotations

# imports
import gettext
import numpy

# local libraries
from nion.data import Calibration
from nion.data import Core
from nion.data import DataAndMetadata
from nion.swift.model import DataStructure
from nion.swift.model import Symbolic
from nion.swift.model import Schema
from nion.swift import Facade
from nion.utils import Registry


_ = gettext.gettext


class EELSBackgroundSubtraction:
    label = _("EELS Background Subtraction")
    inputs = {
        "eels_spectrum_data_item": {"label": _("EELS Spectrum")},
        "background_model": {"label": _("Background Model"), "entity_id": "background_model"},
        "fit_interval_graphics": {"label": _("Fit")},
        }
    outputs = {
        "background": {"label": _("Background")},
        "subtracted": {"label": _("Subtracted")},
    }

    def __init__(self, computation, **kwargs):
        self.computation = computation
        self.__background_xdata = None
        self.__subtracted_xdata = None

    def execute(self, eels_spectrum_data_item, background_model, fit_interval_graphics, **kwargs) -> None:
        try:
            spectrum_xdata = eels_spectrum_data_item.xdata
            assert spectrum_xdata.is_datum_1d
            assert spectrum_xdata.datum_dimensional_calibrations[0].units == "eV"
            eels_spectrum_xdata = spectrum_xdata
            # fit_interval_graphics.interval returns normalized coordinates. create calibrated intervals.
            fit_intervals = list()
            for fit_interval_graphic in fit_interval_graphics:
                fit_interval = Calibration.CalibratedInterval(
                    Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, fit_interval_graphic.interval[0]),
                    Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, fit_interval_graphic.interval[1]))
                fit_intervals.append(fit_interval)
            fit_minimum = min([fit_interval.start.value for fit_interval in fit_intervals])
            signal_interval = Calibration.CalibratedInterval(
                Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, fit_minimum),
                Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, 1.0))
            reference_frame = Calibration.ReferenceFrameAxis(eels_spectrum_xdata.datum_dimensional_calibrations[0],
                                                             eels_spectrum_xdata.datum_dimension_shape[0])
            signal_xdata = Core.get_calibrated_interval_slice(eels_spectrum_xdata, reference_frame, signal_interval)
            background_xdata = None
            subtracted_xdata = None
            if background_model._data_structure.entity:
                entity_id = background_model._data_structure.entity.entity_type.entity_id
                for component in Registry.get_components_by_type("background-model"):
                    if entity_id == component.background_model_id:
                        fit_result = component.fit_background(spectrum_xdata=spectrum_xdata, fit_intervals=fit_intervals, background_interval=signal_interval)
                        background_xdata = fit_result["background_model"]
                        # use 'or' to avoid doing subtraction if subtracted_spectrum already present
                        subtracted_xdata = fit_result.get("subtracted_spectrum", None) or Core.calibrated_subtract_spectrum(spectrum_xdata, background_xdata)
            if background_xdata is None:
                background_xdata = DataAndMetadata.new_data_and_metadata(numpy.zeros_like(signal_xdata.data), intensity_calibration=signal_xdata.intensity_calibration, dimensional_calibrations=signal_xdata.dimensional_calibrations)
            if subtracted_xdata is None:
                subtracted_xdata = DataAndMetadata.new_data_and_metadata(signal_xdata.data, intensity_calibration=signal_xdata.intensity_calibration, dimensional_calibrations=signal_xdata.dimensional_calibrations)
            self.__background_xdata = background_xdata
            self.__subtracted_xdata = subtracted_xdata
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)
            raise

    def commit(self):
        self.computation.set_referenced_xdata("background", self.__background_xdata)
        self.computation.set_referenced_xdata("subtracted", self.__subtracted_xdata)


class EELSMapping:
    label = _("EELS Map")
    inputs = {
        "spectrum_image_data_item": {"label": _("EELS Image")},
        "background_model": {"label": _("Background Model"), "entity_id": "background_model"},
        "fit_interval_graphics": {"label": _("Fit")},
        "signal_interval_graphic": {"label": _("Signal")},
        }
    outputs = {
        "map": {"label": _("EELS Map")},
    }

    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, spectrum_image_data_item: Facade.DataItem, background_model, fit_interval_graphics, signal_interval_graphic):
        try:
            assert spectrum_image_data_item.xdata.is_datum_1d
            assert spectrum_image_data_item.xdata.is_navigable
            assert spectrum_image_data_item.xdata.datum_dimensional_calibrations[0].units == "eV"
            spectrum_image_xdata = spectrum_image_data_item.xdata
            # fit_interval_graphics.interval returns normalized coordinates. create calibrated intervals.
            fit_intervals = list()
            for fit_interval_graphic in fit_interval_graphics:
                fit_interval = Calibration.CalibratedInterval(
                    Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, fit_interval_graphic.interval[0]),
                    Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, fit_interval_graphic.interval[1]))
                fit_intervals.append(fit_interval)
            signal_interval = Calibration.CalibratedInterval(
                Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, signal_interval_graphic.interval[0]),
                Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, signal_interval_graphic.interval[1]))
            mapped_xdata = None
            if background_model._data_structure.entity:
                entity_id = background_model._data_structure.entity.entity_type.entity_id
                for component in Registry.get_components_by_type("background-model"):
                    if entity_id == component.background_model_id:
                        # import time
                        # t0 = time.perf_counter()
                        integrate_result = component.integrate_signal(spectrum_xdata=spectrum_image_xdata, fit_intervals=fit_intervals, signal_interval=signal_interval)
                        # t1 = time.perf_counter()
                        # print(f"{component.background_model_id} {((t1 - t0) * 1000)}ms")
                        mapped_xdata = integrate_result["integrated"]
            if mapped_xdata is None:
                mapped_xdata = DataAndMetadata.new_data_and_metadata(numpy.zeros(spectrum_image_xdata.navigation_dimension_shape), dimensional_calibrations=spectrum_image_xdata.navigation_dimensional_calibrations)
            self.__mapped_xdata = mapped_xdata
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)
            raise

    def commit(self):
        self.computation.set_referenced_xdata("map", self.__mapped_xdata)


async def use_interval_as_background(api: Facade.API_1, window: Facade.DocumentWindow) -> None:
    target_data_item = window.target_data_item
    target_display_item = window.target_display
    target_intervals = [graphic for graphic in target_display_item.selected_graphics if graphic.graphic_type == "interval-graphic"]
    if target_data_item and target_intervals:
        background = api.library.create_data_item(title="{} Background".format(target_data_item.title))
        signal = api.library.create_data_item(title="{} Subtracted".format(target_data_item.title))

        background_model = DataStructure.DataStructure(structure_type="linear_background_model")
        window._document_controller.document_model.append_data_structure(background_model)
        background_model.source = background._data_item

        api.library.create_computation("eels.background_subtraction3",
                                       inputs={
                                           "eels_spectrum_data_item": target_data_item,
                                           "background_model": api._new_api_object(background_model),
                                           "fit_interval_graphics": target_intervals,
                                       },
                                       outputs={
                                           "background": background,
                                           "subtracted": signal}
                                       )
        for target_interval in target_intervals:
            target_interval.graphic_id = "background"
            target_interval.label = _("Background")
        target_display_item._display_item.append_display_data_channel_for_data_item(background._data_item)
        target_display_item._display_item.append_display_data_channel_for_data_item(signal._data_item)
        target_display_item._display_item.display_layers = [
            {"label": "Background", "data_index": 1, "fill_color": "rgba(255, 0, 0, 0.3)"},
            {"label": "Signal", "data_index": 2, "fill_color": "#0F0"},
            {"label": "Data", "data_index": 0, "fill_color": "#1E90FF"},
        ]
        target_display_item._display_item.set_display_property("legend_position", "top-right")


def use_signal_for_map(api, window):
    target_display = window.target_display
    target_graphic = target_display.selected_graphics[0] if target_display and len(target_display.selected_graphics) == 1 else None
    target_interval = target_graphic if target_graphic and target_graphic.graphic_type == "interval-graphic" else None
    if target_display and target_interval:
        target_display_item_data_items = target_display._display_item.data_items
        for computation in api.library._document_model.computations:
            if computation.processing_id == "eels.background_subtraction3":
                if computation.get_input("eels_spectrum_data_item") in target_display_item_data_items and computation.get_output("subtracted") in target_display_item_data_items:
                    eels_spectrum_data_item = computation.get_input("eels_spectrum_data_item")
                    eels_spectrum_data_item = api._new_api_object(eels_spectrum_data_item)
                    fit_interval_graphics = computation.get_input("fit_interval_graphics")
                    fit_interval_graphics = [api._new_api_object(g) for g in fit_interval_graphics]
                    background_model = computation.get_input("background_model")
                    background_model = api._new_api_object(background_model)
                    source_data_items = api.library._document_model.get_source_data_items(eels_spectrum_data_item._data_item)
                    if len(source_data_items) == 1 and source_data_items[0].xdata.is_navigable and source_data_items[0].datum_dimension_count == 1:
                        spectrum_image = api._new_api_object(source_data_items[0])
                        map = api.library.create_data_item_from_data(numpy.zeros(spectrum_image._data_item.xdata.navigation_dimension_shape), title="{} Map".format(spectrum_image.title))
                        signal_interval_graphic = target_interval
                        api.library.create_computation(
                            "eels.mapping3",
                            inputs={
                                "spectrum_image_data_item": spectrum_image,
                                "fit_interval_graphics": fit_interval_graphics,
                                "signal_interval_graphic": signal_interval_graphic,
                                "background_model": background_model,
                            },
                            outputs={
                                "map": map
                            }
                        )
                        window.display_data_item(map)
                    break


def subtract_background_from_signal(api, window):
    window._document_controller.event_loop.create_task(use_interval_as_background(api, window))


Symbolic.register_computation_type("eels.background_subtraction3", EELSBackgroundSubtraction)
Symbolic.register_computation_type("eels.mapping3", EELSMapping)

BackgroundModel = Schema.entity("background_model", None, None, {})


def component_registered(component, component_types):
    if "background-model" in component_types:
        # when a background model is registered, create an empty (for now) entity type, and register it with the data
        # structure so that an entity for use with the UI and computations can be created when the data structure loads.
        background_model_entity = Schema.entity(component.background_model_id, BackgroundModel, None, {})
        DataStructure.DataStructure.register_entity(background_model_entity, entity_name=component.title, entity_package_name=component.package_title)


_component_registered_listener = Registry.listen_component_registered_event(component_registered)

# handle any components that have already been registered.
for component in Registry.get_components_by_type("background-model"):
    component_registered(component, {"background-model"})
