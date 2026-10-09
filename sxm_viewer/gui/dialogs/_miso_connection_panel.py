"""Sugar-connection editor and MISO YAML assistant launcher shared by the
Position monomer and Position coordinates dialogs.

The host dialog provides ``self._instances`` (placed sugar units) and
``self._connections`` (list of SugarConnection), and may override
``_assistant_blocker`` / ``_assistant_has_lipids``.
"""
from __future__ import annotations

from ..._shared import QtCore, QtWidgets
from ...utils.miso_yaml import (
    MISOExportError, SugarConnection, linkage_carbons, validate_connections,
)


class MISOConnectionMixin:
    CONNECTION_HINT = (
        "Connect named units explicitly; branches are allowed. Rebuilding clears "
        "connections and root. YAML export supports sugars only.")

    def _build_connection_ui(self, layout):
        group = QtWidgets.QGroupBox("MISO YAML: sugar connections")
        box = QtWidgets.QVBoxLayout(group)
        hint = QtWidgets.QLabel(self.CONNECTION_HINT)
        hint.setWordWrap(True)
        box.addWidget(hint)
        self.assistant_btn = QtWidgets.QPushButton("Guide me / optional LLM assistant")
        self.assistant_btn.clicked.connect(self._open_miso_assistant)
        box.addWidget(self.assistant_btn)
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
        self.connection_table = QtWidgets.QTableWidget(0, 5)
        self.connection_table.setHorizontalHeaderLabels(
            ["Donor", "Carbon", "Acceptor", "Carbon", "Anomer"])
        self.connection_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.connection_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.connection_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.connection_table.setFixedHeight(110)
        box.addWidget(self.connection_table)
        layout.addWidget(group)
        self._refresh_connection_units()
        return group

    def _reset_connections(self):
        self._connections.clear()
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
        connection = SugarConnection(
            donor, self.donor_carbon_combo.currentText(), acceptor,
            self.acceptor_carbon_combo.currentText(), self.link_anomer_combo.currentText())
        try:
            validate_connections(self._instances, self._connections + [connection])
        except MISOExportError as exc:
            QtWidgets.QMessageBox.warning(self, "MISO connection", str(exc))
            return
        self._connections.append(connection)
        self._fill_connection_table()

    def _remove_connection(self):
        row = self.connection_table.currentRow()
        if row >= 0:
            self._connections.pop(row)
            self.connection_table.removeRow(row)

    def _fill_connection_table(self):
        self.connection_table.setRowCount(len(self._connections))
        for row, link in enumerate(self._connections):
            values = (self._instances[link.donor]["label"], link.donor_carbon,
                      self._instances[link.acceptor]["label"], link.acceptor_carbon, link.anomer)
            for column, value in enumerate(values):
                self.connection_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def _assistant_blocker(self):
        """Message explaining why the assistant cannot open yet, or None."""
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
            self.orientation_combo.currentData(), parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted and dialog.result_choices is not None:
            self._apply_assistant_choices(dialog.result_choices)

    def _apply_assistant_choices(self, choices):
        from ...utils.miso_assistant import validate_choices
        try:
            validate_choices(self._instances, choices)
        except MISOExportError as exc:
            QtWidgets.QMessageBox.warning(self, "MISO assistant", str(exc))
            return
        self._connections = list(choices.connections)
        self._fill_connection_table()
        self.root_combo.setCurrentIndex(self.root_combo.findData(choices.root))
        self.orientation_combo.setCurrentIndex(
            self.orientation_combo.findData(choices.fixed_orientation))
