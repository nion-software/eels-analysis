import functools
import gettext

# ensure background models are registered
from nion.eels_analysis import BackgroundModel

from . import BackgroundSubtraction
from . import ElementalMappingPanel
from . import AlignZLP
from . import ThicknessMap
from . import LiveThickness
from . import LiveZLP

from nion.swift import Facade

_ = gettext.gettext


class MenuExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.eels_analysis"

    def __init__(self, api_broker):
        # grab the api object.
        self.__api = api_broker.get_api(version="~1.0")
        self.__api.application._application.register_menu_handler(self.__build_menus)

        LiveThickness.register_measure_thickness_process(self.__api)
        LiveZLP.register_measure_zlp_process(self.__api)

    def close(self):
        self.__api.application._application.unregister_menu_handler(self.__build_menus)

    def __build_menus(self, document_window):
        api = self.__api
        window = Facade.DocumentWindow(document_window)

        eels_menu = document_window.get_or_create_menu("eels_menu", _("EELS"), "window_menu")

        eels_menu.add_separator()
        eels_menu.add_menu_item(_("Fit Background"), functools.partial(BackgroundSubtraction.subtract_background_from_signal, api, window))
        eels_menu.add_separator()
        eels_menu.add_menu_item(_("Map Signal"), functools.partial(BackgroundSubtraction.use_signal_for_map, api, window))
        eels_menu.add_menu_item(_("Map Thickness"), functools.partial(ThicknessMap.map_thickness, api, window))
        eels_menu.add_separator()
        eels_menu.add_menu_item(_("Align ZLP (max method)"), functools.partial(AlignZLP.align_zlp, api, window))
        eels_menu.add_menu_item(_("Align ZLP (com method)"), functools.partial(AlignZLP.align_zlp_com, api, window))
        eels_menu.add_menu_item(_("Align ZLP (peak fit method)"), functools.partial(AlignZLP.align_zlp_fit, api, window))
        eels_menu.add_separator()
        eels_menu.add_menu_item(_("Show Live Thickness Measurement"), functools.partial(LiveThickness.attach_measure_thickness, api, window))
        eels_menu.add_menu_item(_("Show Live ZLP Measurement"), functools.partial(LiveZLP.attach_measure_zlp, api, window))
