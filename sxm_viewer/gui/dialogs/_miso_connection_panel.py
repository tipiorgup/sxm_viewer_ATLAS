"""Sugar-connection editor and MISO YAML assistant launcher shared by the
Position monomer and Position coordinates dialogs.

The host dialog provides ``self._instances`` (placed sugar units) and
``self._connections`` (list of SugarConnection), and may override
``_assistant_blocker`` / ``_assistant_has_lipids``.
"""
from __future__ import annotations

from pathlib import Path

from ..._shared import QtCore, QtWidgets
from ...utils.miso_yaml import (
    GLYCOSYLATION_TYPES, MISOExportError, PEtNGroup, PeptideBond, SugarConnection,
    default_peptide_order, is_amino_acid, linkage_carbons, validate_connections,
)


class MISOConnectionMixin:
    CONNECTION_HINT = (
        "Connect named units explicitly; branches are allowed. Rebuilding clears "
        "connections and root. YAML export supports sugars only.")
    # Hosts that can export a peptide section set this to True; others stay sugar only.
    SUPPORTS_PEPTIDE = False
    # Hosts that own free image points (Position coordinates) set this to offer PEtN.
    SUPPORTS_PETN = False
    PEPTIDE_HINT = (
        "Join a sugar carbon to an amino acid (Asn, Ser, Thr or Tyr; one sugar per "
        "residue). The chain runs in the order listed below; any residue can be part "
        "of a long peptide.")

    def _build_connection_ui(self, layout):
        group = QtWidgets.QGroupBox("MISO YAML: sugar connections")
        box = QtWidgets.QVBoxLayout(group)
        hint = QtWidgets.QLabel(self.CONNECTION_HINT)
        hint.setWordWrap(True)
        box.addWidget(hint)
        self.assistant_btn = QtWidgets.QPushButton("Guide me / optional LLM assistant")
        self.assistant_btn.clicked.connect(self._open_miso_assistant)
        box.addWidget(self.assistant_btn)
        self.variants_label = QtWidgets.QLabel("")
        self.variants_label.setWordWrap(True)
        self.variants_label.setVisible(False)
        box.addWidget(self.variants_label)
        form = QtWidgets.QFormLayout()
        self.root_combo = QtWidgets.QComboBox()
        self.root_combo.addItem("Choose root monomer...", None)
        form.addRow("Root:", self.root_combo)
        self.orientation_combo = QtWidgets.QComboBox()
        self.orientation_combo.addItem("Choose orientation mode...", None)
        self.orientation_combo.addItem("Keep positioned geometry and rotations", True)
        self.orientation_combo.addItem("Let MISO optimize orientations", False)
        form.addRow("Orientation:", self.orientation_combo)
        self.donor_combo = QtWidgets.QComboBox()
        self.acceptor_combo = QtWidgets.QComboBox()
        self.donor_carbon_combo = QtWidgets.QComboBox()
        self.acceptor_carbon_combo = QtWidgets.QComboBox()
        self.link_anomer_combo = QtWidgets.QComboBox()
        self.link_anomer_combo.addItems(["Choose...", "alpha", "beta"])
        for label, unit, carbon in (
                ("Donor:", self.donor_combo, self.donor_carbon_combo),
                ("Acceptor:", self.acceptor_combo, self.acceptor_carbon_combo)):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(unit, 1)
            row.addWidget(carbon)
            form.addRow(label, row)
        self.link_bond_combo = QtWidgets.QComboBox()
        self.link_bond_combo.addItem("Glycosidic bond", "glycosidic")
        self.link_bond_combo.addItem("Phosphate bridge", "phosphate")
        self.link_bond_combo.setToolTip(
            "Phosphate bridge: sugar carbon, O, P, O, sugar carbon. MISO rotates the donor "
            "toward the acceptor, so the donor can have no other sugar bond and the acceptor "
            "must be in the glycosidic chain.")
        self.link_bond_combo.currentIndexChanged.connect(self._bond_type_changed)
        form.addRow("Bond type:", self.link_bond_combo)
        form.addRow("Linkage anomer:", self.link_anomer_combo)
        box.addLayout(form)
        self.donor_combo.currentIndexChanged.connect(self._refresh_linkage_carbons)
        self.acceptor_combo.currentIndexChanged.connect(self._refresh_linkage_carbons)
        buttons = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add connection")
        add.clicked.connect(self._add_connection)
        remove = QtWidgets.QPushButton("Remove selected connection")
        remove.clicked.connect(self._remove_connection)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        box.addLayout(buttons)
        self.connection_table = QtWidgets.QTableWidget(0, 6)
        self.connection_table.setHorizontalHeaderLabels(
            ["Donor", "Carbon", "Acceptor", "Carbon", "Anomer", "Bond"])
        self.connection_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.connection_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.connection_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.connection_table.setFixedHeight(110)
        box.addWidget(self.connection_table)
        layout.addWidget(group)
        self._peptide_bonds = []
        self._peptide_order = None
        self._petn = []
        if self.SUPPORTS_PEPTIDE:
            self._build_peptide_ui(box)
        self._refresh_connection_units()
        return group

    def _bond_type_changed(self, _index=None):
        phosphate = self.link_bond_combo.currentData() == "phosphate"
        self.link_anomer_combo.setEnabled(not phosphate)
        if phosphate:
            self.link_anomer_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------ PEtN
    def _free_point_indices(self):
        """Picked points that belong to no unit (hosts with free points override this)."""
        return []

    def _petn_point_label(self, index):
        return f"Point {index}"

    def _petn_free_points(self):
        """Allowlisted rows for the assistant: spare points with their coordinates."""
        rows = []
        for index in self._free_point_indices():
            _ox, _oy, x, y, *_rest = self._points[index]
            rows.append({"point": index, "xy_angstrom": [round(float(x), 4), round(float(y), 4)]})
        return rows

    def _petn_sites(self):
        return {(group.sugar, group.sugar_carbon) for group in self._petn}

    def _prune_petn(self):
        """Drop PEtN groups that no longer fit the units or the free points.

        PEtN has no controls in the main panel: it is set in the assistant only.
        """
        if not self.SUPPORTS_PETN:
            return
        free = set(self._free_point_indices())
        self._petn = [g for g in self._petn
                      if 0 <= g.sugar < len(self._instances)
                      and self._instances[g.sugar].get("kind") == "sugar"
                      and g.sugar_carbon in linkage_carbons(self._instances[g.sugar])
                      and g.point in free]

    def _connection_sites(self):
        """Sugar carbons used by sugar bonds and peptide bonds."""
        sites = {(link.donor, link.donor_carbon) for link in self._connections}
        sites |= {(link.acceptor, link.acceptor_carbon) for link in self._connections}
        sites |= {(bond.sugar, bond.sugar_carbon) for bond in self._peptide_bonds}
        return sites

    def _build_peptide_ui(self, box):
        title = QtWidgets.QLabel("<b>Sugar to amino acid bonds</b>")
        box.addWidget(title)
        hint = QtWidgets.QLabel(self.PEPTIDE_HINT)
        hint.setWordWrap(True)
        box.addWidget(hint)
        form = QtWidgets.QFormLayout()
        self.pep_sugar_combo = QtWidgets.QComboBox()
        self.pep_carbon_combo = QtWidgets.QComboBox()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.pep_sugar_combo, 1)
        row.addWidget(self.pep_carbon_combo)
        form.addRow("Sugar:", row)
        self.pep_residue_combo = QtWidgets.QComboBox()
        form.addRow("Residue:", self.pep_residue_combo)
        self.pep_anomer_combo = QtWidgets.QComboBox()
        self.pep_anomer_combo.addItems(["Choose...", "alpha", "beta"])
        form.addRow("Linkage anomer:", self.pep_anomer_combo)
        box.addLayout(form)
        self.pep_sugar_combo.currentIndexChanged.connect(self._refresh_peptide_carbons)
        buttons = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add peptide bond")
        add.clicked.connect(self._add_peptide_bond)
        remove = QtWidgets.QPushButton("Remove selected peptide bond")
        remove.clicked.connect(self._remove_peptide_bond)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        box.addLayout(buttons)
        self.peptide_table = QtWidgets.QTableWidget(0, 4)
        self.peptide_table.setHorizontalHeaderLabels(["Sugar", "Carbon", "Residue", "Anomer"])
        self.peptide_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.peptide_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.peptide_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.peptide_table.setFixedHeight(90)
        box.addWidget(self.peptide_table)
        box.addWidget(QtWidgets.QLabel("Peptide chain order (N-terminus first):"))
        order_row = QtWidgets.QHBoxLayout()
        self.order_list = QtWidgets.QListWidget()
        self.order_list.setFixedHeight(90)
        order_row.addWidget(self.order_list, 1)
        order_buttons = QtWidgets.QVBoxLayout()
        up = QtWidgets.QPushButton("Move up")
        down = QtWidgets.QPushButton("Move down")
        up.clicked.connect(lambda: self._move_residue(-1))
        down.clicked.connect(lambda: self._move_residue(1))
        order_buttons.addWidget(up)
        order_buttons.addWidget(down)
        order_buttons.addStretch()
        order_row.addLayout(order_buttons)
        box.addLayout(order_row)
        from .miso_assistant_dialog import ConnectionSketch
        self.chain_sketch = ConnectionSketch(self._instances)
        self.chain_sketch.setMinimumSize(360, 220)
        box.addWidget(self.chain_sketch)
        self.root_combo.currentIndexChanged.connect(self._refresh_chain_sketch)
        self.orientation_combo.currentIndexChanged.connect(self._refresh_chain_sketch)

    def _refresh_chain_sketch(self):
        """Redraw the plan with the peptide chain arrows (N term to C term)."""
        sketch = getattr(self, "chain_sketch", None)
        if sketch is None:
            return
        sketch.instances = self._instances
        sketch.set_plan(self._connections, self.root_combo.currentData(),
                        self.orientation_combo.currentData(), "Peptide chain",
                        self._peptide_bonds)
        sketch.set_peptide_order(self._current_peptide_order()
                                 if self._residue_indices() else None)

    def _residue_indices(self):
        return [i for i, inst in enumerate(self._instances) if is_amino_acid(inst)]

    def _current_peptide_order(self):
        """Chain order in use: the user's order, else ascending point index."""
        if self._peptide_order is not None and set(self._peptide_order) == set(
                self._residue_indices()):
            return list(self._peptide_order)
        return default_peptide_order(self._instances)

    def _refresh_peptide_carbons(self, _index=None):
        selected = self.pep_carbon_combo.currentText()
        self.pep_carbon_combo.clear()
        index = self.pep_sugar_combo.currentData()
        if index is not None:
            self.pep_carbon_combo.addItems(linkage_carbons(self._instances[index]))
        if selected in [self.pep_carbon_combo.itemText(i)
                        for i in range(self.pep_carbon_combo.count())]:
            self.pep_carbon_combo.setCurrentText(selected)

    def _refresh_peptide_ui(self):
        blockers = [QtCore.QSignalBlocker(self.pep_sugar_combo),
                    QtCore.QSignalBlocker(self.pep_residue_combo)]
        self.pep_sugar_combo.clear()
        self.pep_sugar_combo.addItem("Choose sugar...", None)
        self.pep_residue_combo.clear()
        self.pep_residue_combo.addItem("Choose residue...", None)
        for index, inst in enumerate(self._instances):
            if is_amino_acid(inst):
                if inst["name"] in GLYCOSYLATION_TYPES:
                    self.pep_residue_combo.addItem(inst["label"], index)
            else:
                self.pep_sugar_combo.addItem(inst["label"], index)
        del blockers
        self._refresh_peptide_carbons()
        self._fill_peptide_table()
        self._fill_order_list()

    def _fill_peptide_table(self):
        self.peptide_table.setRowCount(len(self._peptide_bonds))
        for row, bond in enumerate(self._peptide_bonds):
            values = (self._instances[bond.sugar]["label"], bond.sugar_carbon,
                      self._instances[bond.residue]["label"], bond.anomer)
            for column, value in enumerate(values):
                self.peptide_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self._refresh_chain_sketch()

    def _fill_order_list(self):
        self.order_list.clear()
        for number, index in enumerate(self._current_peptide_order(), 1):
            item = QtWidgets.QListWidgetItem(f"{number}. {self._instances[index]['label']}")
            item.setData(QtCore.Qt.UserRole, index)
            self.order_list.addItem(item)
        self._refresh_chain_sketch()

    def _move_residue(self, step):
        row = self.order_list.currentRow()
        order = self._current_peptide_order()
        target = row + step
        if row < 0 or not 0 <= target < len(order):
            return
        order[row], order[target] = order[target], order[row]
        self._peptide_order = order
        self._fill_order_list()
        self.order_list.setCurrentRow(target)

    def _add_peptide_bond(self):
        sugar = self.pep_sugar_combo.currentData()
        residue = self.pep_residue_combo.currentData()
        if sugar is None or residue is None:
            QtWidgets.QMessageBox.warning(
                self, "MISO peptide bond", "Choose both a sugar and a residue.")
            return
        bond = PeptideBond(sugar, self.pep_carbon_combo.currentText(), residue,
                           self.pep_anomer_combo.currentText())
        try:
            validate_connections(self._instances, self._connections,
                                 self._peptide_bonds + [bond], self._used_lipid_sites())
        except MISOExportError as exc:
            QtWidgets.QMessageBox.warning(self, "MISO peptide bond", str(exc))
            return
        self._peptide_bonds.append(bond)
        self._fill_peptide_table()

    def _remove_peptide_bond(self):
        row = self.peptide_table.currentRow()
        if row >= 0:
            self._peptide_bonds.pop(row)
            self.peptide_table.removeRow(row)

    def _used_lipid_sites(self):
        """Sugar carbons already holding a lipid (hosts without lipids return nothing)."""
        sites = getattr(self, "_lipid_sites", None)
        return sites() if sites is not None else set()

    def _used_group_sites(self):
        """Carbons held by lipids or PEtN groups, which a new sugar bond must avoid."""
        return self._used_lipid_sites() | set(self._petn_sites() if self.SUPPORTS_PETN else ())

    # Alternative connection lists proposed by the assistant (empty: ordinary export).
    _variants = ()
    VARIANT_CONFIRM_COUNT = 25

    def _set_variants(self, variants):
        self._variants = [list(v) for v in variants] if len(variants) > 1 else []
        count = len(self._variants)
        self.variants_label.setVisible(bool(count))
        self.variants_label.setText(
            f"Assistant variants: {count}. The table shows variant 1; export writes one "
            "YAML per variant (shared CSV and companion files). Editing the connections "
            "discards the variants." if count else "")

    def _miso_configs(self, out_path):
        """[(yaml_path, config)]: one entry, or one per assistant variant."""
        base = Path(out_path).with_suffix(".yml")
        if len(self._variants) < 2:
            return [(base, self._make_miso_config(out_path))]
        original = self._connections
        results = []
        try:
            for number, variant in enumerate(self._variants, start=1):
                self._connections = list(variant)
                try:
                    config = self._make_miso_config(out_path)
                except MISOExportError as exc:
                    raise MISOExportError(f"Variant {number}: {exc}") from exc
                results.append((base.with_name(f"{base.stem}_v{number}{base.suffix}"), config))
        finally:
            self._connections = original
        return results

    def _reset_connections(self):
        self._set_variants([])
        self._connections.clear()
        self._peptide_bonds = []
        self._peptide_order = None
        self._petn = []
        self.connection_table.setRowCount(0)
        self._refresh_connection_units()

    def _refresh_connection_units(self):
        blockers = [QtCore.QSignalBlocker(combo) for combo in
                    (self.root_combo, self.donor_combo, self.acceptor_combo)]
        for combo, placeholder in (
                (self.root_combo, "Choose root monomer..."),
                (self.donor_combo, "Choose donor..."),
                (self.acceptor_combo, "Choose acceptor...")):
            combo.clear()
            combo.addItem(placeholder, None)
            for index, inst in enumerate(self._instances):
                if inst["kind"] == "sugar":
                    combo.addItem(inst["label"], index)
        del blockers
        self._refresh_linkage_carbons()
        if self.SUPPORTS_PEPTIDE:
            self._refresh_peptide_ui()
        self._prune_petn()
        refresh_lipids = getattr(self, "_refresh_lipid_attach_ui", None)
        if refresh_lipids is not None and hasattr(self, "lipid_attach_sugar"):
            refresh_lipids()

    def _refresh_linkage_carbons(self, _index=None):
        for unit, carbon in ((self.donor_combo, self.donor_carbon_combo),
                             (self.acceptor_combo, self.acceptor_carbon_combo)):
            selected = carbon.currentText()
            carbon.clear()
            index = unit.currentData()
            if index is not None:
                carbon.addItems(linkage_carbons(self._instances[index]))
            if selected in [carbon.itemText(i) for i in range(carbon.count())]:
                carbon.setCurrentText(selected)

    def _add_connection(self):
        donor = self.donor_combo.currentData()
        acceptor = self.acceptor_combo.currentData()
        if donor is None or acceptor is None:
            QtWidgets.QMessageBox.warning(
                self, "MISO connection", "Choose both a donor and an acceptor.")
            return
        bond = self.link_bond_combo.currentData()
        connection = SugarConnection(
            donor, self.donor_carbon_combo.currentText(), acceptor,
            self.acceptor_carbon_combo.currentText(),
            "none" if bond == "phosphate" else self.link_anomer_combo.currentText(), bond)
        try:
            validate_connections(self._instances, self._connections + [connection],
                                 self._peptide_bonds, self._used_group_sites())
        except MISOExportError as exc:
            QtWidgets.QMessageBox.warning(self, "MISO connection", str(exc))
            return
        self._connections.append(connection)
        self._set_variants([])
        self._fill_connection_table()

    def _remove_connection(self):
        row = self.connection_table.currentRow()
        if row >= 0:
            self._connections.pop(row)
            self._set_variants([])
            self.connection_table.removeRow(row)

    def _fill_connection_table(self):
        self.connection_table.setRowCount(len(self._connections))
        for row, link in enumerate(self._connections):
            values = (self._instances[link.donor]["label"], link.donor_carbon,
                      self._instances[link.acceptor]["label"], link.acceptor_carbon,
                      link.anomer if link.bond == "glycosidic" else "n/a",
                      "phosphate" if link.bond == "phosphate" else "glycosidic")
            for column, value in enumerate(values):
                self.connection_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self._refresh_chain_sketch()

    def _assistant_blocker(self):
        """Message explaining why the assistant cannot open yet, or None."""
        if self.SUPPORTS_PEPTIDE:
            if not any(inst["kind"] == "sugar" for inst in self._instances):
                return "Build at least one sugar unit first."
            if any(inst["kind"] not in ("sugar", "aa") for inst in self._instances):
                return "Remove lipid rows before using this assistant."
            lipid_blocker = getattr(self, "_lipid_assistant_blocker", None)
            return lipid_blocker() if lipid_blocker is not None else None
        if not self._instances or any(inst["kind"] != "sugar" for inst in self._instances):
            return ("Build sugar units first. This assistant currently supports sugars only; "
                    "remove amino-acid units and lipid rows before using it.")
        return None

    def _open_miso_assistant(self):
        message = self._assistant_blocker()
        if message:
            QtWidgets.QMessageBox.warning(self, "MISO assistant", message)
            return
        from .miso_assistant_dialog import MISOAssistantDialog
        dialog = MISOAssistantDialog(
            self._instances, self._connections, self.root_combo.currentData(),
            self.orientation_combo.currentData(), parent=self,
            peptide_bonds=self._peptide_bonds,
            peptide_order=self._current_peptide_order() if self._residue_indices() else None,
            lipid_rows=self._lipid_context() if hasattr(self, "_lipid_context") else (),
            free_points=self._petn_free_points() if self.SUPPORTS_PETN else None,
            point_count=len(self._points) if self.SUPPORTS_PETN else None,
            petn=self._petn if self.SUPPORTS_PETN else ())
        if dialog.exec_() == QtWidgets.QDialog.Accepted and dialog.result_choices is not None:
            self._apply_assistant_choices(dialog.result_choices)

    def _apply_assistant_choices(self, choices):
        from dataclasses import replace
        from ...utils.miso_assistant import validate_choices
        try:
            validate_choices(self._instances, choices,
                             self._lipid_context() if hasattr(self, "_lipid_context") else (),
                             len(self._points) if self.SUPPORTS_PETN else None)
        except MISOExportError as exc:
            QtWidgets.QMessageBox.warning(self, "MISO assistant", str(exc))
            return
        variants = list(getattr(choices, "alternatives", None) or [])
        if len(variants) > 1:
            try:
                for variant in variants:
                    validate_choices(self._instances, replace(choices, connections=variant),
                                     self._lipid_context() if hasattr(self, "_lipid_context")
                                     else (), len(self._points) if self.SUPPORTS_PETN else None)
            except MISOExportError as exc:
                QtWidgets.QMessageBox.warning(self, "MISO assistant", str(exc))
                return
            if len(variants) >= self.VARIANT_CONFIRM_COUNT:
                answer = QtWidgets.QMessageBox.question(
                    self, "MISO assistant",
                    f"The assistant proposes {len(variants)} variants, so export will write "
                    f"{len(variants)} YAML files. Continue?")
                if answer != QtWidgets.QMessageBox.Yes:
                    return
        self._connections = list(choices.connections)
        self._set_variants(variants)
        if self.SUPPORTS_PETN:
            self._petn = list(choices.petn)
            self._prune_petn()
        if choices.lipid_attachments and hasattr(self, "_apply_lipid_attachments"):
            self._apply_lipid_attachments(choices.lipid_attachments)
        self._peptide_bonds = list(choices.peptide_bonds)
        self._peptide_order = (list(choices.peptide_order)
                               if choices.peptide_order is not None else None)
        if self.SUPPORTS_PEPTIDE:
            self._fill_peptide_table()
            self._fill_order_list()
        self._fill_connection_table()
        self.root_combo.setCurrentIndex(self.root_combo.findData(choices.root))
        self.orientation_combo.setCurrentIndex(
            self.orientation_combo.findData(choices.fixed_orientation))
