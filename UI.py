# standard libraries
import gettext

# third party libraries
# None

# local libraries
from nion.swift import Application

_ = gettext.gettext


def processing_extract_signal(document_controller):
    from nion.swift.model import DataItem
    from nion.swift.model import DocumentModel

    display_specifier = document_controller.selected_display_specifier

    fit1_region = DocumentModel.DocumentModel.make_region("fit1", "interval", params={"label": _("Fit1"), "interval": (0.2, 0.3)})
    fit2_region = DocumentModel.DocumentModel.make_region("fit2", "interval", params={"label": _("Fit2"), "interval": (0.3, 0.4)})
    signal_region = DocumentModel.DocumentModel.make_region("signal", "interval", params={"label": _("Signal"), "interval": (0.4, 0.5)})
    src = DocumentModel.DocumentModel.make_source(display_specifier.data_item, None, "src", _("Source"), regions=[fit1_region, fit2_region, signal_region])
    data_item = document_controller.document_model.make_data_item_with_computation("extract_signal_from_polynomial_background({src}, signal.interval, (fit1.interval, fit2.interval))", [src], [],
                                                                                   _("Background Subtracted"))
    if data_item:
        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(data_item)
        document_controller.display_data_item(new_display_specifier)
        return data_item
    return None


def build_menus(document_controller):
    document_controller.processing_menu.add_menu_item(_("Extract Signal"), lambda: processing_extract_signal(document_controller))


Application.app.register_menu_handler(build_menus)  # called on import to make the menu entry for this plugin
