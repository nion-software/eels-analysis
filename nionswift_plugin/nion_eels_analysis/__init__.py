import dataclasses
import gettext
import pkgutil

# ensure background models are registered
from nion.eels_analysis import BackgroundModel
from nion.eels_analysis import PeakModel

from . import EELSMenuActions
from . import ElementalMappingPanel
from . import LiveThickness
from . import LiveZLP
from . import PeakFitting

from nion.swift import DocumentController
from nion.swift.model import PlugInManager
from nion.ui import Window
from nion.utils import Registry

_ = gettext.gettext


class MenuExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.eels_analysis"

    def __init__(self, api_broker: PlugInManager.APIBroker) -> None:
        # grab the api object.
        self.__api = api_broker.get_api(version="~1.0")

        # register the actions once for every window, and before the menu handler, which builds the menu of each existing window from them.
        for eels_menu_action_type in EELSMenuActions.eels_menu_action_types:
            Window.register_action(eels_menu_action_type())
        self.__api.application._application.register_menu_handler(self.__build_menus)

        LiveThickness.register_measure_thickness_process(self.__api)
        LiveZLP.register_measure_zlp_process(self.__api)

        xml_bytes = pkgutil.get_data(__name__, "resources/color_maps/sqe_bgyw.xml")
        assert xml_bytes is not None
        xml_str = xml_bytes.decode("utf-8")

        @dataclasses.dataclass(frozen=True)
        class ColorMapDescription:
            color_map_id: str
            name: str
            xml_str: str

        # the color map component is kept so that close can unregister it.
        self.__color_map_description = ColorMapDescription("areels-high-contrast", _("AREELS High Contrast"), xml_str)
        Registry.register_component(self.__color_map_description, {"color-map-description"})

    def close(self) -> None:
        self.__api.application._application.unregister_menu_handler(self.__build_menus)
        for eels_menu_action_type in EELSMenuActions.eels_menu_action_types:
            Window.unregister_action(eels_menu_action_type.action_id)
        Registry.unregister_component(self.__color_map_description, {"color-map-description"})

    def __build_menus(self, document_window: DocumentController.DocumentController) -> None:
        eels_menu = document_window.get_or_create_menu("eels_menu", _("EELS"), "window_menu")

        document_window.build_menu(eels_menu, [
            {"type": "separator"},
            {"type": "item", "action_id": EELSMenuActions.FitBackgroundAction.action_id},
            # {"type": "item", "action_id": "eels.fit_zero_loss_peak"},  # Fit Zero Loss Peak, calling PeakFitting.fit_zero_loss_peak
            {"type": "separator"},
            # {"type": "item", "action_id": "eels.subtract_background"},  # Subtract Background, calling BackgroundSubtraction.subtract_background
            # {"type": "separator"},
            {"type": "item", "action_id": EELSMenuActions.MapSignalAction.action_id},
            {"type": "item", "action_id": EELSMenuActions.MapThicknessAction.action_id},
            {"type": "separator"},
            {"type": "item", "action_id": EELSMenuActions.AlignZLPMaxAction.action_id},
            {"type": "item", "action_id": EELSMenuActions.AlignZLPComAction.action_id},
            {"type": "item", "action_id": EELSMenuActions.AlignZLPFitAction.action_id},
            {"type": "separator"},
            {"type": "item", "action_id": EELSMenuActions.ShowLiveThicknessAction.action_id},
            {"type": "item", "action_id": EELSMenuActions.ShowLiveZLPAction.action_id},
            {"type": "separator"},
            {"type": "item", "action_id": EELSMenuActions.CalibrateSpectrumAction.action_id},
            {"type": "separator"},
            {"type": "item", "action_id": EELSMenuActions.MeasureTemperatureAction.action_id},
            {"type": "separator"},
        ])
