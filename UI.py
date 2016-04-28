# standard libraries
import copy
import gettext

# third party libraries
# None

# local libraries
import_ok = False
try:
    from nion.swift import Application
    from nion.swift.model import DataItem
    from nion.swift.model import DocumentModel
    from nion.swift.model import Graphics
    import_ok = True
except ImportError:
    pass

_ = gettext.gettext


def processing_extract_signal(document_controller):
    display_specifier = document_controller.selected_display_specifier

    fit_region = DocumentModel.DocumentModel.make_region("fit", "interval", params={"label": _("Fit"), "interval": (0.2, 0.3)})
    signal_region = DocumentModel.DocumentModel.make_region("signal", "interval", params={"label": _("Signal"), "interval": (0.4, 0.5)})
    src = DocumentModel.DocumentModel.make_source(display_specifier.data_item, None, "src", _("Source"), regions=[fit_region, signal_region])
    data_item = document_controller.document_model.make_data_item_with_computation("vstack((extract_signal_from_polynomial_background({src}, signal.interval, (fit.interval, )), {src})", [src], [],
                                                                                   _("Background Subtracted"))
    if data_item:
        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(data_item)
        document_controller.display_data_item(new_display_specifier)
        return data_item
    return None


def processing_subtract_linear_background(document_controller):
    display_specifier = document_controller.selected_display_specifier
    fit_region = DocumentModel.DocumentModel.make_region("fit", "interval", params={"label": _("Fit"), "interval": (0.2, 0.3)})
    src = DocumentModel.DocumentModel.make_source(display_specifier.data_item, None, "src", _("Source"), regions=[fit_region, ])
    data_item = document_controller.document_model.make_data_item_with_computation("vstack((subtract_linear_background({src}, fit.interval, (0, 1)), {src}))", [src], [],
                                                                                   _("Linear Background Subtracted"))
    if data_item:
        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(data_item)
        document_controller.display_data_item(new_display_specifier)
        return data_item
    return None


def show_color_channels(document_controller):
    display_specifier = document_controller.selected_display_specifier
    display = display_specifier.display
    if display:
        names = (_("Red"), _("Green"), _("Blue"))
        for r in range(1, 4):
            region = Graphics.ChannelGraphic()
            region.label = names[r - 1]
            region.position = r / 4
            region.is_shape_locked = True
            display.add_graphic(region)


def filter_channel(document_controller):
    document_model = document_controller.document_model
    display_specifier = document_controller.selected_display_specifier
    data_item = display_specifier.data_item
    if data_item:
        display = data_item.maybe_data_source.displays[0]
        selected_graphics = display.selected_graphics
        selected_graphic = selected_graphics[0] if len(selected_graphics) == 1 else None
        selected_region = None
        for region in display.graphics:
            if region == selected_graphic:
                selected_region = region
                break
        if selected_region:
            src_data_items = document_model.get_source_data_items(data_item)
            if len(src_data_items) == 1:
                pick_data_item = src_data_items[0]
                src_data_items = document_model.get_source_data_items(pick_data_item)
                if len(src_data_items) == 1:
                    src_data_item = src_data_items[0]
                    fit_region = copy.deepcopy(data_item.maybe_data_source.computation.variables[1])
                    src = DocumentModel.DocumentModel.make_source(src_data_item, None, "src", _("Source"), use_display_data=False)
                    script = "sum(stacked_subtract_linear_background(src.data, fit.interval, signal.interval))"
                    new_data_item = document_model.make_data_item_with_computation(script, [src], [], _("Mapped"))
                    computation = new_data_item.maybe_data_source.computation
                    computation.create_object("signal", document_model.get_object_specifier(selected_region), label=_("Signal"), cascade_delete=True)
                    computation.add_variable(fit_region)
                    if new_data_item:
                        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(new_data_item)
                        document_controller.display_data_item(new_display_specifier)
                        return new_data_item
    return None


def build_menus(document_controller):
    document_controller.processing_menu.add_menu_item(_("Subtract Linear Background"), lambda: processing_subtract_linear_background(document_controller))
    document_controller.processing_menu.add_menu_item(_("Extract Signal"), lambda: processing_extract_signal(document_controller))
    document_controller.processing_menu.add_menu_item(_("Show Color Channels"), lambda: show_color_channels(document_controller))
    document_controller.processing_menu.add_menu_item(_("Filter Channel"), lambda: filter_channel(document_controller))


if import_ok and Application.app is not None:
    Application.app.register_menu_handler(build_menus)  # called on import to make the menu entry for this plugin
