import h5py
import numpy
import pathlib
import tempfile
import typing
import unittest

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift import Application
from nion.swift import Facade
from nion.swift.model import DataItem
from nion.swift.test import TestContext
from nion.ui import TestUI

from .. import AlignZLP


Facade.initialize()


class TestBackgroundSubtraction(unittest.TestCase):

    def setUp(self) -> None:
        self._test_setup = TestContext.TestSetup(set_global=True)

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)

    def test_calibrate_spectrum_for_single_spectrum(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data = numpy.zeros((100,), dtype=numpy.float32)
            data[10] = 1
            intensity_calibration = Calibration.Calibration(units="~")
            dimensional_calibrations = [Calibration.Calibration(scale=1.0, units="eV")]
            data_descriptor = DataAndMetadata.DataDescriptor(is_sequence=False, collection_dimension_count=0, datum_dimension_count=1)
            xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
            data_item = DataItem.new_data_item(xdata)
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            document_controller.select_display_items_in_data_panel([display_item])
            document_controller.data_panel_focused()
            api = Facade.get_api("~1.0", "~1.0")
            dialog = AlignZLP._calibrate_spectrum(api, api.application.document_windows[0])
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(api.library.data_items))
            # calibrate_spectrum should create two graphics, offset and scale
            self.assertEqual(2, len(api.library.data_items[0].graphics))
            # It should find the peak at 10 and set the calibration offset accordingly
            self.assertAlmostEqual(data_item.dimensional_calibrations[0].offset, -10.5)
            # Move the scale graphic and check that the calibration changed accordingly
            offset_graphic = api.library.data_items[0].graphics[0]
            scale_graphic = api.library.data_items[0].graphics[1]
            position_diff = scale_graphic.position - offset_graphic.position
            scale_graphic.position = offset_graphic.position + 0.5 * position_diff
            self.assertAlmostEqual(data_item.dimensional_calibrations[0].scale, 2.0)
            # Move the offset graphic and check that the calibration offset and the scale graphic have moved
            position_diff = scale_graphic.position - offset_graphic.position
            offset_graphic.position += 0.1
            self.assertAlmostEqual(position_diff, scale_graphic.position - offset_graphic.position)
            self.assertAlmostEqual(data_item.dimensional_calibrations[0].offset, -41)
            # Closing the dialog should remove the graphics
            dialog.request_close()
            self.assertEqual(0, len(api.library.data_items[0].graphics))
            # Test cleanup
            document_model.remove_data_item(data_item)
            self.assertEqual(0, len(document_model.data_items))
            self.assertEqual(0, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))

    def test_align_zlp_shifts_spectrum_image_to_reference_in_middle_row(self) -> None:
        # each spectrum has one peak, at a position and of a height of its own. aligning moves each peak to the position
        # of the reference spectrum's peak, which is in the middle row.
        peak_positions = [8 + (5 * i) % 11 for i in range(12)]
        ref_index = 6
        data = numpy.zeros((3, 4, 32), dtype=numpy.float32)
        expected_data = numpy.zeros_like(data)
        for i, peak_position in enumerate(peak_positions):
            data[i // 4, i % 4, peak_position] = 10 + i
            expected_data[i // 4, i % 4, peak_positions[ref_index]] = 10 + i
        expected_shifts = numpy.reshape(numpy.subtract(peak_positions, peak_positions[ref_index]), (3, 4)).astype(numpy.float32)
        dimensional_calibrations = [Calibration.Calibration(), Calibration.Calibration(), Calibration.Calibration(scale=1.0, units="eV")]
        data_descriptor = DataAndMetadata.DataDescriptor(False, 2, 1)
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(pathlib.Path(directory) / "data.h5", "w") as file:
                # nionswift hands out data stored in HDF5 as an h5py dataset, or a data handle over one.
                for label, source_data in (("in memory", data), ("in an HDF5 file", file.create_dataset("data", data=data))):
                    with self.subTest(label):
                        src_xdata = DataAndMetadata.DataAndMetadata(source_data, data_shape_and_dtype=(data.shape, data.dtype),
                                                                    dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
                        aligned_xdata, shift_xdata = AlignZLP.align_zlp_xdata(src_xdata, method="max", ref_index=ref_index)
                        assert aligned_xdata
                        assert shift_xdata
                        self.assertTrue(numpy.array_equal(expected_data, aligned_xdata.data), "an aligned peak is not at the reference position, or not of its spectrum's height")
                        self.assertTrue(numpy.array_equal(expected_shifts, shift_xdata.data), "a shift is not the distance between a spectrum's peak and the reference peak")

    def test_align_zlp_menu_aligns_sequence_to_displayed_frame(self) -> None:
        # each frame has its peak at a channel of its own. the menu action aligns the frames to the displayed frame.
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            peak_positions = [6, 12, 20, 25]
            data = numpy.zeros((4, 32), dtype=numpy.float32)
            data[range(4), peak_positions] = 1
            dimensional_calibrations = [Calibration.Calibration(), Calibration.Calibration(scale=1.0, units="eV")]
            data_descriptor = DataAndMetadata.DataDescriptor(True, 0, 1)
            data_item = DataItem.new_data_item(DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor))
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            display_item.display_data_channel.sequence_index = 2
            document_controller.select_display_items_in_data_panel([display_item])
            document_controller.data_panel_focused()
            document_controller.periodic()
            api = Facade.get_api("~1.0", "~1.0")
            AlignZLP.align_zlp(api, api.application.document_windows[0])
            aligned_data = document_model.data_items[-1].data
            self.assertEqual([peak_positions[2]] * 4, numpy.argmax(aligned_data, axis=-1).tolist(), "a frame's peak is not at the peak of the displayed frame")
