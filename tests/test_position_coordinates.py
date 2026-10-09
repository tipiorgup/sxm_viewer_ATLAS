import csv
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import yaml

from sxm_viewer._shared import QtCore, QtWidgets
from sxm_viewer.gui.dialogs.position_coordinates_dialogs import PositionCoordinatesDialog
from sxm_viewer.gui.dialogs.position_monomer_dialogs import _SugarLookupWorker
from sxm_viewer.utils.miso_yaml import SugarConnection
from sxm_viewer.utils.sugar_lookup import (
    SugarConnectionError, SugarResult, parse_sugar_description,
)

from test_miso_yaml import sugar

GLUCOSE = "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O"
MODULE = "sxm_viewer.gui.dialogs.position_coordinates_dialogs"


def fake_image(self):
    self._img = np.arange(400, dtype=float).reshape(20, 20)
    self._px = (0.1, 0.1)
    self._scan_dir = "up"
    self._sxm_path = Path("scan.sxm")


def template():
    inst = sugar()
    return {"conf_name": inst["conf_name"], "rel": inst["rel"],
            "atom_types": inst["atom_types"], "bonds": inst["bonds"],
            "rigid": inst["rigid"]}


class PositionCoordinatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.viewer = QtWidgets.QWidget()
        with patch.object(PositionCoordinatesDialog, "_load_demo_image", fake_image):
            self.dialog = PositionCoordinatesDialog(self.viewer)
        self.info = patch.object(QtWidgets.QMessageBox, "information").start()
        self.warning = patch.object(QtWidgets.QMessageBox, "warning").start()
        self.addCleanup(patch.stopall)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.out = Path(self.folder.name) / "scan_positions.csv"
        self.dialog.csv_le.setText(str(self.out))

    def tearDown(self):
        QtCore.QThreadPool.globalInstance().waitForDone(2000)
        self.dialog.close()
        self.dialog.deleteLater()
        self.viewer.deleteLater()
        self.app.processEvents()

    def click(self, col, row):
        self.dialog._on_canvas_click(
            SimpleNamespace(inaxes=True, button=1, xdata=col, ydata=row))

    def set_sugar(self, row, text, name=""):
        self.dialog.table.item(row, PositionCoordinatesDialog.COL_SUGAR).setText(text)
        self.dialog.table.item(row, PositionCoordinatesDialog.COL_NAME).setText(name)

    def build(self):
        with patch(f"{MODULE}.import_monomer_engine"), \
                patch(f"{MODULE}.build_sugar_template", return_value=template()) as builder:
            self.dialog._build_monomers()
        return builder

    def test_basic_export_without_sugars_is_unchanged(self):
        self.click(2, 3)
        self.click(5, 7)
        self.dialog._export_csv()
        with open(self.out, newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["Point", "Original_X", "Original_Y", "X (Angstrom)",
                                   "Y (Angstrom)", "Height", "Z (Angstrom)"])
        self.assertEqual([r[:3] for r in rows[1:]], [["0", "2", "16"], ["1", "5", "12"]])
        self.assertEqual(float(rows[1][5]), self.dialog._img[16, 2])
        self.assertTrue(self.out.with_suffix(".npz").exists())
        self.assertTrue(self.out.with_suffix(".png").exists())
        self.assertFalse(self.out.with_suffix(".yml").exists())
        self.assertFalse(Path(f"{self.out.with_suffix('')}_orientations.csv").exists())
        self.warning.assert_not_called()

    def test_typed_entries_survive_new_points_and_clear_last(self):
        self.click(1, 1)
        self.set_sugar(0, "chair KDO", "KDO")
        self.dialog.table.cellWidget(0, PositionCoordinatesDialog.COL_RING).setCurrentText("chair")
        self.click(4, 4)
        self.click(6, 6)
        self.dialog._clear_last()
        self.assertEqual(self.dialog.table.rowCount(), 2)
        self.assertEqual(self.dialog._sugar_entries(), [(0, "KDO", "chair KDO", "chair", "Any")])
        self.assertEqual(self.dialog.count_lbl.text(), "Points: 2")

    def test_build_and_export_use_point_indices_with_unassigned_points(self):
        for p in range(3):
            self.click(2 + p, 3 + p)
        self.set_sugar(0, GLUCOSE, "Glc")
        self.set_sugar(2, GLUCOSE)          # blank name -> automatic name
        builder = self.build()
        builder.assert_called_once()        # same sugar built once
        labels = [inst["label"] for inst in self.dialog._instances]
        self.assertEqual(labels, ["Glc.0", "Sugar1.2"])
        self.assertEqual(self.dialog.table.item(2, PositionCoordinatesDialog.COL_NAME).text(),
                         "Sugar1")
        inst = self.dialog._instances[1]
        self.assertEqual(inst["com"][:2], list(self.dialog._points[2][2:4]))
        self.assertIn("Built 2 sugar unit(s)", self.dialog.build_status.text())

        self.dialog.table.setCurrentCell(2, 0)
        self.dialog.spin["rz"].setValue(30.0)
        self.assertEqual(inst["euler"], [0.0, 0.0, 30.0])
        self.dialog._connections = [SugarConnection(1, "C1", 0, "C4", "beta")]
        self.dialog._fill_connection_table()
        self.dialog.root_combo.setCurrentIndex(self.dialog.root_combo.findData(0))
        self.dialog.orientation_combo.setCurrentIndex(
            self.dialog.orientation_combo.findData(True))
        self.dialog._export_csv()
        self.warning.assert_not_called()

        config = yaml.safe_load(self.out.with_suffix(".yml").read_text(encoding="utf-8"))
        self.assertEqual(config["experimental_positions"], {"Glc": [0], "Sugar1": [2]})
        self.assertEqual(config["root_mol"], "Glc_0")
        self.assertEqual(config["glycosidic_bonds"][0][:5],
                         ["Sugar1_2", "C1", "Glc_0", "C4", "beta"])
        self.assertEqual(config["circle_input_path"], str(self.out.resolve()))
        with open(self.out, newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)
        with open(config["orientation_csv_path"], newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["Point"] for r in rows], ["0", "2"])
        self.assertAlmostEqual(float(rows[1]["R01"]), -0.5)
        with open(config["monomer_data_path"], "rb") as handle:
            self.assertEqual(set(pickle.load(handle)), {"Glc", "Sugar1"})
        self.assertEqual(self.viewer.last_monomer_yaml,
                         str(self.out.with_suffix(".yml").resolve()))

    def test_same_name_for_different_sugars_is_rejected(self):
        self.click(1, 1)
        self.click(2, 2)
        self.set_sugar(0, GLUCOSE, "X")
        self.set_sugar(1, "OCC1OC(O)CC1O", "X")
        builder = self.build()
        builder.assert_not_called()
        self.assertIn("different sugars", self.warning.call_args.args[2])

    def test_changed_entries_block_yaml_and_assistant(self):
        self.click(1, 1)
        self.set_sugar(0, GLUCOSE, "Glc")
        self.build()
        self.assertIsNone(self.dialog._assistant_blocker())
        self.set_sugar(0, GLUCOSE, "Glucose")
        self.assertIn("changed", self.dialog.build_status.text())
        self.assertIn("Build monomers", self.dialog._assistant_blocker())
        with patch.object(QtWidgets.QMessageBox, "question",
                          return_value=QtWidgets.QMessageBox.No) as question:
            self.dialog._export_csv()
        question.assert_called_once()
        self.assertFalse(self.out.exists())
        with patch.object(QtWidgets.QMessageBox, "question",
                          return_value=QtWidgets.QMessageBox.Yes):
            self.dialog._export_csv()
        self.assertTrue(self.out.exists())
        self.assertFalse(self.out.with_suffix(".yml").exists())

    def test_assistant_needs_built_sugars(self):
        self.assertIn("Build sugar units first", self.dialog._assistant_blocker())

    def test_name_lookup_fills_row_and_builds(self):
        self.click(1, 1)
        self.set_sugar(0, "D chair 4C1 glucose")
        request = parse_sugar_description("D chair 4C1 glucose")
        with patch(f"{MODULE}.import_monomer_engine"), \
                patch(f"{MODULE}.build_sugar_template", return_value=template()):
            self.dialog._on_sugar_lookup_finished(
                [(0, request, SugarResult(GLUCOSE, 5793, "glucose"))], "", "")
        self.assertEqual(self.dialog.table.item(0, PositionCoordinatesDialog.COL_SUGAR).text(),
                         GLUCOSE)
        self.assertEqual(self.dialog.table.cellWidget(
            0, PositionCoordinatesDialog.COL_RING).currentText(), "chair_4C1")
        self.assertEqual([i["label"] for i in self.dialog._instances],
                         [f"{request.name}.0"])

    def test_no_internet_warns_and_keeps_entries(self):
        self.click(1, 1)
        self.set_sugar(0, "chair KDO")
        worker = _SugarLookupWorker([(0, parse_sugar_description("chair KDO"))])
        worker.signals.finished.connect(self.dialog._on_sugar_lookup_finished)
        with patch("sxm_viewer.gui.dialogs.position_monomer_dialogs.lookup_sugar",
                   side_effect=SugarConnectionError("Offline.")):
            worker.run()
        title, message = self.warning.call_args.args[1:3]
        self.assertEqual(title, "No internet connection")
        self.assertIn("manually", message)
        self.assertEqual(self.dialog.table.item(0, PositionCoordinatesDialog.COL_SUGAR).text(),
                         "chair KDO")
        self.assertEqual(self.dialog._instances, [])


if __name__ == "__main__":
    unittest.main()
