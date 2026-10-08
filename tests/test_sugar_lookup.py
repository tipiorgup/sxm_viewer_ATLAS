import io
import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from sxm_viewer._shared import QtCore, QtWidgets
from sxm_viewer.gui.dialogs.position_monomer_dialogs import (
    PositionMonomerDialog, _SugarLookupWorker,
)
from sxm_viewer.utils.sugar_lookup import (
    SugarConnectionError, SugarLookupError, SugarResult,
    lookup_sugar, parse_sugar_description,
)


GLUCOSE = "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O"


class SugarDescriptionTests(unittest.TestCase):
    def test_user_examples(self):
        kdo = parse_sugar_description("chair KDO")
        self.assertEqual((kdo.name, kdo.query, kdo.ring), ("KDO", "KDO", "chair"))
        glucose = parse_sugar_description("D chair 4C1 glucose")
        self.assertEqual((glucose.query, glucose.ring, glucose.anomer),
                         ("D-glucose", "chair_4C1", "Any"))

    def test_anomer_and_configuration(self):
        for text in ("beta-D-glucose", "D beta glucose", "Beta D glucose"):
            request = parse_sugar_description(text)
            self.assertEqual((request.query, request.anomer), ("beta-D-glucose", "beta"))
        self.assertEqual(parse_sugar_description("L 1C4 glucose").query, "L-glucose")
        self.assertEqual(parse_sugar_description("half-chair glucose").ring, "half")

    def test_table_selections_are_preserved_and_used_in_query(self):
        request = parse_sugar_description("D glucose", "boat", "alpha")
        self.assertEqual((request.query, request.ring), ("alpha-D-glucose", "boat"))
        self.assertEqual(parse_sugar_description("chair glucose", "chair_4C1").ring,
                         "chair_4C1")
        self.assertEqual(parse_sugar_description("4C1 glucose", "chair").ring,
                         "chair_4C1")

    def test_conflicts_and_missing_name_are_reported(self):
        for text in ("D L glucose", "alpha beta glucose", "4C1 1C4 glucose",
                     "boat 4C1 glucose", "chair boat glucose", "D chair 4C1", ""):
            with self.subTest(text=text), self.assertRaises(SugarLookupError):
                parse_sugar_description(text)
        with self.assertRaises(SugarLookupError):
            parse_sugar_description("chair glucose", "boat")
        with self.assertRaises(SugarLookupError):
            parse_sugar_description("alpha glucose", anomer="beta")

    def test_hyphenated_names_survive(self):
        self.assertEqual(parse_sugar_description("D 2-deoxy-glucose").query,
                         "D-2-deoxy-glucose")


class PubChemTests(unittest.TestCase):
    def _response(self, record):
        return io.BytesIO(json.dumps(
            {"PropertyTable": {"Properties": [record]}}).encode())

    @patch("sxm_viewer.utils.sugar_lookup.urlopen")
    def test_stereochemical_smiles_and_url_encoding(self, open_url):
        open_url.return_value = self._response(
            {"SMILES": GLUCOSE, "CID": 5793, "IUPACName": "glucose"})
        result = lookup_sugar("D glucose/sugar")
        self.assertEqual(result, SugarResult(GLUCOSE, 5793, "glucose"))
        args, kwargs = open_url.call_args
        self.assertIn("D%20glucose%2Fsugar", args[0].full_url)
        self.assertEqual(kwargs["timeout"], 10)

    @patch("sxm_viewer.utils.sugar_lookup.urlopen")
    def test_legacy_isomeric_smiles(self, open_url):
        open_url.return_value = self._response({"IsomericSMILES": GLUCOSE, "CID": 5793})
        self.assertEqual(lookup_sugar("glucose").smiles, GLUCOSE)

    @patch("sxm_viewer.utils.sugar_lookup.urlopen")
    def test_connection_errors(self, open_url):
        for error in (URLError("DNS unavailable"), TimeoutError("timeout"),
                      ConnectionResetError("connection reset")):
            open_url.side_effect = error
            with self.subTest(error=error), self.assertRaises(SugarConnectionError) as cm:
                lookup_sugar("glucose")
            self.assertIn("given manually", str(cm.exception))

    @patch("sxm_viewer.utils.sugar_lookup.urlopen")
    def test_http_errors_are_not_mislabeled_as_offline(self, open_url):
        for code in (404, 429, 503):
            open_url.side_effect = HTTPError("url", code, "error", {}, None)
            with self.subTest(code=code), self.assertRaises(SugarLookupError) as cm:
                lookup_sugar("missing")
            self.assertNotIsInstance(cm.exception, SugarConnectionError)
            if code == 404:
                self.assertIn("did not find", str(cm.exception))

    @patch("sxm_viewer.utils.sugar_lookup.urlopen")
    def test_malformed_responses_are_reported(self, open_url):
        payloads = (b"not json", b"{}", b'{"PropertyTable":{"Properties":[]}}',
                    json.dumps({"PropertyTable": {"Properties": [
                        {"ConnectivitySMILES": "CO", "CID": 1}]}}).encode(),
                    json.dumps({"PropertyTable": {"Properties": [
                        {"SMILES": GLUCOSE, "CID": "wrong"}]}}).encode())
        for payload in payloads:
            open_url.return_value = io.BytesIO(payload)
            with self.subTest(payload=payload), self.assertRaises(SugarLookupError):
                lookup_sugar("glucose")


class PositionMonomerLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.viewer = QtWidgets.QWidget()
        image = {"img": None, "px": (1, 1), "scan_dir": None,
                 "sxm_path": None, "error": "test image"}
        with patch("sxm_viewer.gui.dialogs.position_monomer_dialogs.load_demo_image",
                   return_value=image):
            self.dialog = PositionMonomerDialog(self.viewer)
        self.dialog._img = np.zeros((2, 2))
        self.dialog._build_resolved_monomers = Mock()
        self.warning = patch.object(QtWidgets.QMessageBox, "warning").start()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        QtCore.QThreadPool.globalInstance().waitForDone(2000)
        self.dialog.close()
        self.dialog.deleteLater()
        self.viewer.deleteLater()
        self.app.processEvents()

    def test_manual_smiles_and_amino_acids_do_not_use_internet(self):
        self.dialog.table.item(0, 2).setText(GLUCOSE)
        self.dialog._add_table_row("Amino acid", "Asn")
        with patch("sxm_viewer.gui.dialogs.position_monomer_dialogs.lookup_sugar") as lookup:
            self.dialog._build_monomers()
        lookup.assert_not_called()
        defs = self.dialog._build_resolved_monomers.call_args.args[0]
        self.assertEqual([d["kind"] for d in defs], ["sugar", "aa"])
        self.assertEqual(defs[0]["smiles"], GLUCOSE)

    def test_named_instance_labels_are_used_in_list_and_exports(self):
        self.dialog.table.item(0, 1).setText("KDO")
        self.dialog.table.item(0, 2).setText(GLUCOSE)
        self.dialog._add_table_row("Sugar", GLUCOSE)
        self.dialog.table.item(1, 1).setText("glucose")
        self.dialog.table.cellWidget(1, 5).setValue(2)
        self.dialog._add_table_row("Sugar", GLUCOSE)
        self.dialog.table.item(2, 1).setText("KDO")
        self.dialog._add_table_row("Amino acid", "Asn")
        self.dialog._add_table_row("Sugar", GLUCOSE)
        template = {"conf_name": "chair_4C1_beta_rank1", "rel": np.zeros((1, 3)),
                    "atom_types": ["C"], "bonds": []}
        with patch("sxm_viewer.gui.dialogs.position_monomer_dialogs.import_monomer_engine"), \
                patch.object(self.dialog, "_build_sugar_template", return_value=template), \
                patch.object(self.dialog, "_build_aa_template", return_value=template), \
                patch.object(self.dialog, "_redraw_overlay"), \
                patch.object(QtWidgets.QMessageBox, "information"):
            PositionMonomerDialog._build_resolved_monomers(
                self.dialog, self.dialog._collect_defs())
        expected = ["KDO.1.1", "glucose.2.1", "glucose.2.2", "KDO.3.1",
                    "Asn.4.1", "Sugar5.5.1"]
        self.assertEqual([inst["label"] for inst in self.dialog._instances], expected)
        self.assertEqual([self.dialog.inst_list.item(i).text().split("  [")[0]
                          for i in range(self.dialog.inst_list.count())], expected)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "monomers.csv"
            paths = self.dialog._export_miso_inputs(str(output))
            for path in paths:
                with open(path, newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([row["Instance"] for row in rows], expected)
                self.assertEqual([int(row["Point"]) for row in rows], list(range(6)))
            self.dialog.csv_le.setText(str(output))
            with patch.object(self.dialog, "_export_miso_inputs", return_value=[]), \
                    patch.object(self.dialog, "_export_monomer_pickle", return_value=[]), \
                    patch.object(self.dialog, "_export_png"), \
                    patch.object(QtWidgets.QMessageBox, "information"):
                self.dialog._export_csv()
            with open(output, newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["Instance"] for row in rows], expected)

    def test_async_name_resolution_continues_build_and_preserves_custom_name(self):
        self.dialog.table.item(0, 2).setText("D chair 4C1 glucose")
        self.dialog.table.item(0, 1).setText("Glc")
        self.dialog.table.cellWidget(0, 5).setValue(3)
        loop = QtCore.QEventLoop()
        self.dialog._build_resolved_monomers.side_effect = lambda defs: loop.quit()
        timeout = QtCore.QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        with patch("sxm_viewer.gui.dialogs.position_monomer_dialogs.lookup_sugar",
                   return_value=SugarResult(GLUCOSE, 5793, "glucose")) as lookup:
            self.dialog._build_monomers()
            self.assertFalse(self.dialog.table.isEnabled())
            timeout.start(2000)
            loop.exec_()
            timeout.stop()
        lookup.assert_called_once_with("D-glucose")
        self.dialog._build_resolved_monomers.assert_called_once()
        defs = self.dialog._build_resolved_monomers.call_args.args[0]
        self.assertEqual((defs[0]["name"], defs[0]["smiles"], defs[0]["ring"], defs[0]["copies"]),
                         ("Glc", GLUCOSE, "chair_4C1", 3))
        self.assertTrue(self.dialog.table.isEnabled())
        self.warning.assert_not_called()

    def test_lookup_fills_blank_name_and_anomer(self):
        request = parse_sugar_description("beta-D-glucose")
        self.dialog._on_sugar_lookup_finished(
            [(0, request, SugarResult(GLUCOSE, 5793, "glucose"))], "", "")
        self.assertEqual(self.dialog.table.item(0, 1).text(), "glucose")
        self.assertEqual(self.dialog.table.cellWidget(0, 4).currentText(), "beta")
        self.assertIn("5793", self.dialog.table.item(0, 2).toolTip())

    def test_offline_warning_preserves_inputs_and_placements(self):
        self.dialog.table.item(0, 2).setText("chair KDO")
        self.dialog._instances = [{"label": "existing placement"}]
        worker = _SugarLookupWorker([(0, parse_sugar_description("chair KDO"))])
        worker.signals.finished.connect(self.dialog._on_sugar_lookup_finished)
        with patch("sxm_viewer.gui.dialogs.position_monomer_dialogs.lookup_sugar",
                   side_effect=SugarConnectionError("No internet; enter SMILES manually.")):
            worker.run()
        self.assertEqual(self.dialog.table.item(0, 2).text(), "chair KDO")
        self.assertEqual(self.dialog._instances, [{"label": "existing placement"}])
        self.dialog._build_resolved_monomers.assert_not_called()
        self.assertEqual(self.warning.call_args.args[1], "No internet connection")
        self.assertTrue(self.dialog.table.isEnabled())

    def test_worker_caches_repeated_queries(self):
        worker = _SugarLookupWorker([
            (0, parse_sugar_description("chair glucose")),
            (1, parse_sugar_description("boat glucose"))])
        finished = Mock()
        worker.signals.finished.connect(finished)
        with patch("sxm_viewer.gui.dialogs.position_monomer_dialogs.lookup_sugar",
                   return_value=SugarResult(GLUCOSE, 5793, "glucose")) as lookup:
            worker.run()
        lookup.assert_called_once_with("glucose")
        self.assertEqual(len(finished.call_args.args[0]), 2)

    def test_failed_batch_does_not_apply_partial_results(self):
        self.dialog.table.item(0, 2).setText("glucose")
        self.dialog._add_table_row("Sugar", "unknown sugar")
        worker = _SugarLookupWorker([
            (0, parse_sugar_description("glucose")),
            (1, parse_sugar_description("unknown sugar"))])
        worker.signals.finished.connect(self.dialog._on_sugar_lookup_finished)
        with patch("sxm_viewer.gui.dialogs.position_monomer_dialogs.lookup_sugar",
                   side_effect=[SugarResult(GLUCOSE, 5793, ""),
                                SugarLookupError("PubChem did not find this sugar.")]):
            worker.run()
        self.assertEqual(self.dialog.table.item(0, 2).text(), "glucose")
        self.assertEqual(self.dialog.table.item(1, 2).text(), "unknown sugar")
        self.dialog._build_resolved_monomers.assert_not_called()
        self.assertEqual(self.warning.call_args.args[1], "Sugar lookup")

    def test_conflicting_details_stop_before_lookup(self):
        self.dialog.table.item(0, 2).setText("chair KDO")
        self.dialog.table.cellWidget(0, 3).setCurrentText("boat")
        with patch("sxm_viewer.gui.dialogs.position_monomer_dialogs.lookup_sugar") as lookup:
            self.dialog._build_monomers()
        lookup.assert_not_called()
        self.dialog._build_resolved_monomers.assert_not_called()
        self.assertEqual(self.warning.call_args.args[1], "Sugar description")

    def test_closed_dialog_does_not_apply_pending_results(self):
        self.dialog.reject()
        self.dialog._on_sugar_lookup_finished(
            [(0, parse_sugar_description("glucose"), SugarResult(GLUCOSE, 5793, ""))], "", "")
        self.assertEqual(self.dialog.table.item(0, 2).text(), "")
        self.dialog._build_resolved_monomers.assert_not_called()

    def test_invalid_resolved_smiles_do_not_replace_inputs(self):
        self.dialog.table.item(0, 2).setText("glucose")
        self.dialog._on_sugar_lookup_finished(
            [(0, parse_sugar_description("glucose"), SugarResult("invalid", 5793, ""))], "", "")
        self.assertEqual(self.dialog.table.item(0, 2).text(), "glucose")
        self.dialog._build_resolved_monomers.assert_not_called()
        self.warning.assert_called_once()

    def test_specific_chair_filter_selects_lowest_energy_matching_conformer(self):
        def conformer(ring):
            return {"puckering_type": ring, "coordinates": [[0, 0, 0]],
                    "COM": [0, 0, 0], "atom_types": ["C"]}
        engine = SimpleNamespace(
            generate_monomer_conformers=Mock(return_value={
                "opposite": conformer("chair_1C4"),
                "matching": conformer("chair_4C1")}),
            extract_rigid_monomer_data=Mock(return_value={}))
        template = self.dialog._build_sugar_template(engine, GLUCOSE, "chair_4C1", "Any")
        self.assertEqual(template["conf_name"], "matching")
        self.assertEqual(engine.generate_monomer_conformers.call_args.kwargs["known_ring_type"],
                         "chair")
        engine.generate_monomer_conformers.return_value = {
            "opposite": conformer("chair_1C4")}
        self.assertIsNone(
            self.dialog._build_sugar_template(engine, GLUCOSE, "chair_4C1", "Any"))


if __name__ == "__main__":
    unittest.main()
