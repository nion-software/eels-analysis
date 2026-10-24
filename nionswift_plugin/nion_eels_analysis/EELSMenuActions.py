import gettext
import logging
import typing

from . import AlignZLP
from . import BackgroundSubtraction
from . import LiveThickness
from . import LiveZLP
from . import ThicknessMap
from . import Thermometry

from nion.swift import DocumentController
from nion.swift import Facade
from nion.ui import Window

_ = gettext.gettext


class EELSMenuAction(Window.Action):
    """An item of the EELS menu, run on the window which the action context describes.

    A subclass gives the action id and name, and implements run_menu_item. If the item needs a particular selection, it
    also gives instructions and implements is_selection_suitable. Choosing the item without a suitable selection shows
    the instructions in a dialog; running it from code logs them instead.

    One instance of each subclass is shared by every window, so the window comes from the context each time.
    """

    instructions: str = str()

    def execute(self, context: Window.ActionContext) -> Window.ActionResult:
        if not self.__is_context_suitable(context):
            logging.debug("%s: %s", self.action_name, self.instructions)
            return Window.ActionResult(Window.ActionStatus.CANCELLED)
        return self._run_in_context_window(context)

    def invoke(self, context: Window.ActionContext) -> Window.ActionResult:
        if not self.__is_context_suitable(context):
            if context.window and context.window.app:
                context.window.app.show_ok_dialog(self.action_name, self.instructions)
            return Window.ActionResult(Window.ActionStatus.CANCELLED)
        return self._run_in_context_window(context)

    def is_selection_suitable(self, context: DocumentController.DocumentController.ActionContext) -> bool:
        """Return whether the selection in the context suits the item.

        It runs on the main thread just before the item, so it looks only at metadata and doesn't load any data.
        """
        return True

    def __is_context_suitable(self, context: Window.ActionContext) -> bool:
        # without a document window there's no selection to check, and _run_in_context_window cancels the item anyway.
        if not isinstance(context, DocumentController.DocumentController.ActionContext):
            return True
        return self.is_selection_suitable(context)

    def _run_in_context_window(self, context: Window.ActionContext) -> Window.ActionResult:
        # each item works on the selection of a document window. code may run the action without a window, such as
        # through the application's execute_action, and then there is nothing for the item to work on.
        document_controller = context.window
        if not isinstance(document_controller, DocumentController.DocumentController):
            return Window.ActionResult(Window.ActionStatus.CANCELLED)
        self.run_menu_item(Facade.get_api("~1.0"), Facade.DocumentWindow(document_controller))
        return Window.ActionResult(Window.ActionStatus.FINISHED)

    def run_menu_item(self, api: Facade.API_1, window: Facade.DocumentWindow) -> None:
        """Run the menu item on the window."""
        raise NotImplementedError()


class FitBackgroundAction(EELSMenuAction):
    action_id = "eels.fit_background"
    action_name = _("Fit Background")
    instructions = _("Add an interval to a spectrum and select it to fit a background in that interval.")

    def is_selection_suitable(self, context: DocumentController.DocumentController.ActionContext) -> bool:
        return BackgroundSubtraction.can_fit_background(context.display_item)

    def run_menu_item(self, api: Facade.API_1, window: Facade.DocumentWindow) -> None:
        BackgroundSubtraction.subtract_background_from_signal(api, window)


class MapSignalAction(EELSMenuAction):
    action_id = "eels.map_signal"
    action_name = _("Map Signal")
    instructions = _("Select one interval on a spectrum picked from a spectrum image, after using Fit Background on it, to map the signal in that interval.")

    def is_selection_suitable(self, context: DocumentController.DocumentController.ActionContext) -> bool:
        return BackgroundSubtraction.can_map_signal(context.model, context.display_item)

    def run_menu_item(self, api: Facade.API_1, window: Facade.DocumentWindow) -> None:
        BackgroundSubtraction.use_signal_for_map(api, window)


class MapThicknessAction(EELSMenuAction):
    action_id = "eels.map_thickness"
    action_name = _("Map Thickness")

    def run_menu_item(self, api: Facade.API_1, window: Facade.DocumentWindow) -> None:
        ThicknessMap.map_thickness(api, window)


class AlignZLPMaxAction(EELSMenuAction):
    action_id = "eels.align_zlp_max"
    action_name = _("Align ZLP (max method)")

    def run_menu_item(self, api: Facade.API_1, window: Facade.DocumentWindow) -> None:
        AlignZLP.align_zlp(api, window)


class AlignZLPComAction(EELSMenuAction):
    action_id = "eels.align_zlp_com"
    action_name = _("Align ZLP (com method)")

    def run_menu_item(self, api: Facade.API_1, window: Facade.DocumentWindow) -> None:
        AlignZLP.align_zlp_com(api, window)


class AlignZLPFitAction(EELSMenuAction):
    action_id = "eels.align_zlp_fit"
    action_name = _("Align ZLP (peak fit method)")

    def run_menu_item(self, api: Facade.API_1, window: Facade.DocumentWindow) -> None:
        AlignZLP.align_zlp_fit(api, window)


class ShowLiveThicknessAction(EELSMenuAction):
    action_id = "eels.show_live_thickness"
    action_name = _("Show Live Thickness Measurement")

    def run_menu_item(self, api: Facade.API_1, window: Facade.DocumentWindow) -> None:
        LiveThickness.attach_measure_thickness(api, window)


class ShowLiveZLPAction(EELSMenuAction):
    action_id = "eels.show_live_zlp"
    action_name = _("Show Live ZLP Measurement")

    def run_menu_item(self, api: Facade.API_1, window: Facade.DocumentWindow) -> None:
        LiveZLP.attach_measure_zlp(api, window)


class CalibrateSpectrumAction(EELSMenuAction):
    action_id = "eels.calibrate_spectrum"
    action_name = _("Calibrate Spectrum")

    # the item opens a dialog, so it runs only when a person invokes it.
    def execute(self, context: Window.ActionContext) -> Window.ActionResult:
        raise NotImplementedError()

    def run_menu_item(self, api: Facade.API_1, window: Facade.DocumentWindow) -> None:
        AlignZLP.calibrate_spectrum(api, window)


class MeasureTemperatureAction(EELSMenuAction):
    action_id = "eels.measure_temperature"
    action_name = _("Measure Temperature")
    instructions = _("Select two spectra, in the Data panel or in two display panels, to measure the temperature.")

    def is_selection_suitable(self, context: DocumentController.DocumentController.ActionContext) -> bool:
        # the measurement compares two different spectra, so two displays of the same spectrum don't count.
        display_items = context.display_items
        if len(display_items) != 2:
            return False
        data_items = [display_item.data_item for display_item in display_items]
        if data_items[0] is data_items[1]:
            return False
        return all(data_item is not None and data_item.is_data_1d for data_item in data_items)

    def run_menu_item(self, api: Facade.API_1, window: Facade.DocumentWindow) -> None:
        Thermometry.measure_temperature(api, window)


# the actions of the EELS menu, which the menu extension registers while it is open.
eels_menu_action_types: typing.Sequence[type[EELSMenuAction]] = (
    FitBackgroundAction,
    MapSignalAction,
    MapThicknessAction,
    AlignZLPMaxAction,
    AlignZLPComAction,
    AlignZLPFitAction,
    ShowLiveThicknessAction,
    ShowLiveZLPAction,
    CalibrateSpectrumAction,
    MeasureTemperatureAction,
)
