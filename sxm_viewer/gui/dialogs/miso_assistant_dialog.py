from __future__ import annotations

from copy import deepcopy
import json

import math

from ..._shared import QtCore, QtGui, QtWidgets
from ...utils.miso_assistant import (
    AssistantChoices, AssistantError, PROVIDERS, ServiceConfig, delete_api_key,
    molecular_context, parse_reply, read_api_key, request_advice, save_api_key,
    validate_choices,
)
from ...utils.miso_yaml import MISOExportError, SugarConnection, linkage_carbons, validate_connections


class _AdviceSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(str, str)


class _AdviceWorker(QtCore.QRunnable):
    def __init__(self, config, key, context, history):
        super().__init__()
        self.config, self.key = config, key
        self.context, self.history = context, history
        self.signals = _AdviceSignals()

    def run(self):
        try:
            reply = request_advice(self.config, self.key, self.context, self.history)
        except AssistantError as exc:
            self.signals.finished.emit("", str(exc))
        else:
            self.signals.finished.emit(reply, "")
        finally:
            self.key = ""


class ConnectionSketch(QtWidgets.QWidget):
    """Skeletal plan of placed sugars and donor-to-acceptor bonds, drawn top-down."""

    def __init__(self, instances, parent=None):
        super().__init__(parent)
        self.instances = instances
        self.connections = []
        self.root = None
        self.fixed_orientation = None
        self.caption = ""
        self.setMinimumSize(420, 260)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_plan(self, connections, root=None, fixed_orientation=None, caption=""):
        self.connections = list(connections)
        self.root = root
        self.fixed_orientation = fixed_orientation
        self.caption = caption
        self.update()

    ELEMENT_COLORS = {"C": "#303030", "O": "#e53935", "N": "#1e88e5",
                      "S": "#c9a800", "P": "#fb8c00"}

    @staticmethod
    def unit_geometry(inst):
        """Positioned heavy atoms (x, y in Å) and linkable carbons of a unit.

        Returns None when the unit has no built structure, so it is drawn as a dot.
        """
        rel = inst.get("rel")
        types = inst.get("atom_types")
        if rel is None or not types:
            return None
        from .position_monomer_dialogs import euler_to_matrix
        import numpy as np
        coords = np.asarray(rel, dtype=float) @ euler_to_matrix(*inst["euler"]).T
        coords = coords + np.asarray(inst["com"], dtype=float)
        heavy = {i: (float(coords[i][0]), float(coords[i][1]), sym)
                 for i, sym in enumerate(types) if sym != "H"}
        bonds = [(a, b) for a, b in inst.get("bonds") or () if a in heavy and b in heavy]
        carbon_map = ((inst.get("rigid") or {}).get(inst.get("conf_name"), {})
                      .get("carbon_map", {}))
        carbons = {name: heavy[carbon_map[name]][:2] for name in linkage_carbons(inst)
                   if carbon_map.get(name) in heavy}
        return {"atoms": heavy, "bonds": bonds, "carbons": carbons}

    def _mapper(self, geometries):
        """Å -> widget transform fitting every unit; y points up as in the STM image."""
        xs, ys = [], []
        for inst, geo in zip(self.instances, geometries):
            if geo:
                xs += [x for x, _y, _s in geo["atoms"].values()]
                ys += [y for _x, y, _s in geo["atoms"].values()]
            xs.append(float(inst["com"][0]))
            ys.append(float(inst["com"][1]))
        margin, top = 48.0, 34.0
        width = max(1.0, self.width() - 2 * margin)
        height = max(1.0, self.height() - margin - top - 40.0)
        span = max(max(xs) - min(xs), max(ys) - min(ys), 4.0 if any(geometries) else 1e-9)
        scale = min(width, height) / span
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        mid_x, mid_y = self.width() / 2, top + 24.0 + height / 2
        return lambda x, y: QtCore.QPointF(mid_x + (x - cx) * scale, mid_y - (y - cy) * scale)

    def node_points(self):
        """Widget coordinates of every unit centre (COM)."""
        if not self.instances:
            return []
        to_widget = self._mapper([self.unit_geometry(inst) for inst in self.instances])
        return [to_widget(float(inst["com"][0]), float(inst["com"][1]))
                for inst in self.instances]

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtCore.Qt.white)
        painter.setPen(QtGui.QPen(QtGui.QColor("#b0b0b0")))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.setPen(QtCore.Qt.black)
        header = self.caption or "Connection plan"
        if self.fixed_orientation is not None:
            header += " | " + ("keep positioned rotations" if self.fixed_orientation
                               else "MISO optimizes orientations")
        painter.drawText(QtCore.QRectF(8, 4, self.width() - 16, 20),
                         QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, header)
        if not self.instances:
            painter.end()
            return
        geometries = [self.unit_geometry(inst) for inst in self.instances]
        to_widget = self._mapper(geometries)
        centres = [to_widget(float(inst["com"][0]), float(inst["com"][1]))
                   for inst in self.instances]
        valid = [link for link in self.connections
                 if 0 <= link.donor < len(centres) and 0 <= link.acceptor < len(centres)]
        linked = {i for link in valid for i in (link.donor, link.acceptor)}
        used = {(link.donor, link.donor_carbon) for link in valid}
        used |= {(link.acceptor, link.acceptor_carbon) for link in valid}

        label_tops = []
        for index, (inst, geo, centre) in enumerate(zip(self.instances, geometries, centres)):
            is_root = index == self.root
            color = QtGui.QColor("steelblue" if index in linked or is_root else "#a0a0a0")
            if geo is None:
                points = {}
                extent = 11.0
                painter.setPen(QtGui.QPen(color.darker(130), 1.5))
                painter.setBrush(color)
                painter.drawEllipse(centre, extent, extent)
            else:
                points = {i: to_widget(x, y) for i, (x, y, _s) in geo["atoms"].items()}
                extent = max([math.hypot(p.x() - centre.x(), p.y() - centre.y())
                              for p in points.values()] + [8.0]) + 8.0
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 28))
                painter.drawEllipse(centre, extent, extent)
                painter.setPen(QtGui.QPen(color.darker(120), 2.0))
                for a, b in geo["bonds"]:
                    painter.drawLine(points[a], points[b])
                for i, point in points.items():
                    sym = geo["atoms"][i][2]
                    atom_color = QtGui.QColor(self.ELEMENT_COLORS.get(sym, "#8e24aa"))
                    painter.setPen(QtGui.QPen(QtCore.Qt.white, 0.8))
                    painter.setBrush(atom_color)
                    painter.drawEllipse(point, 3.2, 3.2)
            if is_root:
                painter.setPen(QtGui.QPen(QtGui.QColor("#d4a017"), 3))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(centre, extent + 4, extent + 4)
            label_tops.append(centre.y() - extent - 4)

        for link in valid:
            start = self._site_point(geometries, centres, to_widget, link.donor, link.donor_carbon)
            end = self._site_point(geometries, centres, to_widget, link.acceptor, link.acceptor_carbon)
            radius = 11.0 if geometries[link.donor] is None else 5.0
            anomer = {"alpha": "\u03b1", "beta": "\u03b2"}.get(link.anomer, link.anomer)
            # Carbon names are already shown on drawn structures.
            label = (anomer if geometries[link.donor] and geometries[link.acceptor]
                     else f"{link.donor_carbon}\u2192{link.acceptor_carbon} {anomer}")
            self._draw_bond(painter, start, end, radius, label)

        small = QtGui.QFont(painter.font())
        small.setPointSizeF(max(6.5, small.pointSizeF() - 1.5))
        for index, (geo, centre) in enumerate(zip(geometries, centres)):
            if not geo:
                continue
            for name, (x, y) in geo["carbons"].items():
                point = to_widget(x, y)
                dx, dy = point.x() - centre.x(), point.y() - centre.y()
                length = math.hypot(dx, dy) or 1.0
                is_used = (index, name) in used
                small.setBold(is_used)
                painter.setFont(small)
                box = painter.fontMetrics().boundingRect(name).adjusted(-2, 0, 2, 0)
                box.moveCenter(QtCore.QPoint(int(point.x() + dx / length * 12),
                                             int(point.y() + dy / length * 12)))
                painter.setPen(QtGui.QPen(QtGui.QColor("#d97706"), 1.2) if is_used else QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(255, 243, 205, 235) if is_used
                                 else QtGui.QColor(255, 255, 255, 200))
                painter.drawRoundedRect(QtCore.QRectF(box), 2, 2)
                painter.setPen(QtGui.QColor("#9a4a00") if is_used else QtGui.QColor("#1f4e79"))
                painter.drawText(box, QtCore.Qt.AlignCenter, name)
        small.setBold(False)

        font = QtGui.QFont(painter.font())
        font.setPointSizeF(small.pointSizeF() + 1.5)
        for index, (inst, centre, label_top) in enumerate(zip(self.instances, centres, label_tops)):
            is_root = index == self.root
            font.setBold(True)
            painter.setFont(font)
            text = inst["label"]
            box = painter.fontMetrics().boundingRect(text).adjusted(-3, -1, 3, 1)
            box.moveCenter(QtCore.QPoint(int(centre.x()), int(label_top - box.height() / 2)))
            box.moveLeft(max(box.left(), 2))
            box.moveRight(min(box.right(), self.width() - 4))
            box.moveTop(max(box.top(), 26))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(255, 224, 130, 235) if is_root else QtGui.QColor(255, 255, 255, 210))
            painter.drawRoundedRect(QtCore.QRectF(box), 3, 3)
            painter.setPen(QtCore.Qt.black)
            painter.drawText(box, QtCore.Qt.AlignCenter, text)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#606060"))
        legend = ("Drawn as positioned (top view). C1, C2\u2026: carbons that can form a linkage. "
                  "Arrows: donor \u2192 acceptor (\u03b1/\u03b2). Gold: root. Grey: not connected yet.")
        painter.drawText(QtCore.QRectF(8, self.height() - 38, self.width() - 16, 34),
                         QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap, legend)
        painter.end()

    @staticmethod
    def _site_point(geometries, centres, to_widget, index, carbon):
        geo = geometries[index]
        if geo and carbon in geo["carbons"]:
            return to_widget(*geo["carbons"][carbon])
        return centres[index]

    @staticmethod
    def _draw_bond(painter, start, end, radius, label):
        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = math.hypot(dx, dy)
        if length <= 2 * radius:
            return
        ux, uy = dx / length, dy / length
        tail = QtCore.QPointF(start.x() + ux * radius, start.y() + uy * radius)
        tip = QtCore.QPointF(end.x() - ux * radius, end.y() - uy * radius)
        color = QtGui.QColor("#d97706")
        painter.setPen(QtGui.QPen(color, 2.5))
        painter.drawLine(tail, tip)
        size = 10.0
        left = QtCore.QPointF(tip.x() - ux * size - uy * size * 0.5, tip.y() - uy * size + ux * size * 0.5)
        right = QtCore.QPointF(tip.x() - ux * size + uy * size * 0.5, tip.y() - uy * size - ux * size * 0.5)
        painter.setBrush(color)
        painter.drawPolygon(QtGui.QPolygonF([tip, left, right]))
        mid = QtCore.QPointF((tail.x() + tip.x()) / 2, (tail.y() + tip.y()) / 2)
        box = painter.fontMetrics().boundingRect(label).adjusted(-3, -1, 3, 1)
        visible = math.hypot(tip.x() - tail.x(), tip.y() - tail.y())
        if box.width() > 0.6 * visible:
            nx, ny = (-uy, ux) if ux >= 0 else (uy, -ux)
            if ny < 0:
                nx, ny = -nx, -ny
            shift = box.height() / 2 + 6 + abs(nx) * box.width() / 2
            mid = QtCore.QPointF(mid.x() + nx * shift, mid.y() + ny * shift)
        box.moveCenter(mid.toPoint())
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 248, 230, 230))
        painter.drawRoundedRect(QtCore.QRectF(box), 3, 3)
        painter.setPen(color.darker(140))
        painter.drawText(box, QtCore.Qt.AlignCenter, label)


class ServiceSettingsPanel(QtWidgets.QWidget):
    """Hosted LLM provider, model and API key settings shared by the MISO assistants."""

    status = QtCore.pyqtSignal(str)
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QtCore.QSettings("SXMViewer", "MISOAssistant")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.settings_toggle = QtWidgets.QToolButton()
        self.settings_toggle.setText("Service settings")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setChecked(True)
        self.settings_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.settings_toggle.setArrowType(QtCore.Qt.DownArrow)
        layout.addWidget(self.settings_toggle)
        self.settings_box = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(self.settings_box)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_toggle.toggled.connect(self._toggle)
        form = QtWidgets.QFormLayout()
        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItems(PROVIDERS)
        self.endpoint_edit = QtWidgets.QLineEdit()
        self.endpoint_edit.setPlaceholderText("Complete HTTPS API request URL from your institution")
        self.model_edit = QtWidgets.QLineEdit()
        self.model_edit.setPlaceholderText("Model identifier from your provider or administrator")
        self.key_edit = QtWidgets.QLineEdit()
        self.key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.key_edit.setPlaceholderText("API key (never written to YAML or ordinary settings)")
        self.auth_combo = QtWidgets.QComboBox()
        self.auth_combo.addItem("Bearer token", "Authorization")
        self.auth_combo.addItem("Institutional api-key header", "api-key")
        for label, control in (("Provider:", self.provider_combo), ("Request URL:", self.endpoint_edit),
                               ("Model:", self.model_edit), ("API key:", self.key_edit),
                               ("Authentication:", self.auth_combo)):
            form.addRow(label, control)
        settings_layout.addLayout(form)
        self.remember_key = QtWidgets.QCheckBox("Remember API key in Windows Credential Manager")
        self.remember_key.setChecked(True)
        settings_layout.addWidget(self.remember_key)
        buttons = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("Save configuration")
        load = QtWidgets.QPushButton("Load saved key")
        forget = QtWidgets.QPushButton("Remove saved key")
        save.clicked.connect(self._save_settings)
        load.clicked.connect(self._load_key)
        forget.clicked.connect(self._forget_key)
        for button in (save, load, forget):
            buttons.addWidget(button)
        settings_layout.addLayout(buttons)
        layout.addWidget(self.settings_box)
        self.provider_combo.currentTextChanged.connect(self._provider_changed)
        self.endpoint_edit.textEdited.connect(self._endpoint_changed)
        self.auth_combo.currentIndexChanged.connect(self._endpoint_changed)
        self.model_edit.textEdited.connect(lambda _text: self.changed.emit())
        self._provider_changed(self.provider_combo.currentText())

    def _toggle(self, shown):
        self.settings_box.setVisible(shown)
        self.settings_toggle.setArrowType(QtCore.Qt.DownArrow if shown else QtCore.Qt.RightArrow)

    def _provider_changed(self, provider):
        self.endpoint_edit.setText(PROVIDERS[provider])
        self.auth_combo.setEnabled(provider != "Anthropic")
        self.key_edit.clear()
        self.changed.emit()

    def _endpoint_changed(self, _value=None):
        self.key_edit.clear()
        self.changed.emit()

    def service_config(self):
        return ServiceConfig(self.provider_combo.currentText(), self.endpoint_edit.text().strip(),
                             self.model_edit.text().strip(), self.auth_combo.currentData())

    def api_key(self):
        return self.key_edit.text()

    def ready_config(self):
        """Validated config with a key, or raise AssistantError (and show the settings)."""
        config = self.service_config()
        try:
            config.validate()
            if not self.key_edit.text().strip():
                raise AssistantError("Enter an API key or click Load saved key.")
        except AssistantError:
            self.settings_toggle.setChecked(True)
            raise
        return config

    def load_settings(self):
        provider = self.settings.value("provider", "OpenAI")
        if provider not in PROVIDERS:
            self.status.emit("Saved provider is unsupported. Configure the online service again.")
            return
        self.provider_combo.setCurrentText(provider)
        self.endpoint_edit.setText(self.settings.value("endpoint", PROVIDERS[provider]))
        self.model_edit.setText(self.settings.value("model", ""))
        self.auth_combo.setCurrentIndex(max(0, self.auth_combo.findData(
            self.settings.value("auth_header", "Authorization"))))
        self.settings_toggle.setChecked(not self.model_edit.text().strip())
        self.changed.emit()

    def _save_settings(self):
        config = self.service_config()
        try:
            config.validate()
            if self.remember_key.isChecked():
                save_api_key(config, self.key_edit.text())
        except AssistantError as exc:
            self._error(str(exc))
            return
        for key, value in (("provider", config.provider), ("endpoint", config.endpoint),
                           ("model", config.model), ("auth_header", config.auth_header)):
            self.settings.setValue(key, value)
        self.settings.sync()
        if self.settings.status() != QtCore.QSettings.NoError:
            self._error("Could not save provider settings. Your key may already be stored in Credential Manager.")
            return
        self.status.emit(
            "Configuration saved. Use Load saved key next time." if self.remember_key.isChecked()
            else "Provider settings saved; the API key is for this session only.")

    def _load_key(self):
        try:
            config = self.service_config()
            config.validate()
            key = read_api_key(config)
        except AssistantError as exc:
            self._error(str(exc))
            return
        self.key_edit.setText(key)
        self.status.emit("Saved key loaded." if key else
                         "No saved key found for this service. Enter one or use the offline options.")

    def _forget_key(self):
        try:
            config = self.service_config()
            config.validate()
            delete_api_key(config)
        except AssistantError as exc:
            self._error(str(exc))
            return
        self.key_edit.clear()
        self.changed.emit()
        self.status.emit("Saved API key removed.")

    def _error(self, message):
        QtWidgets.QMessageBox.warning(self.window(), "LLM service", message)


class MISOAssistantDialog(QtWidgets.QDialog):
    """An offline question-by-question guide, with opt-in hosted proposals."""

    def __init__(self, instances, connections, root, mode, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MISO YAML assistant")
        self.resize(1100, 760)
        self.instances = deepcopy(instances)
        self.connections = list(connections)
        self.context = molecular_context(self.instances, connections, root, mode)
        self.result_choices = None
        self._online_choices = None
        self._history = []
        self._consent_config = None
        self._worker = None
        self._closed = False
        self.finished.connect(self._on_closed)
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_offline(root, mode), "Offline guide")
        self.tabs.addTab(self._build_online(), "Online LLM (optional)")
        layout.addWidget(self.tabs)
        close = QtWidgets.QPushButton("Close without applying")
        close.clicked.connect(self.reject)
        layout.addWidget(close)
        self._load_settings()

    def _on_closed(self, _result):
        self._closed = True
        self.key_edit.clear()

    def _unit_combo(self):
        combo = QtWidgets.QComboBox()
        combo.addItem("Choose a unit...", None)
        for index, inst in enumerate(self.instances):
            combo.addItem(inst["label"], index)
        return combo

    def _build_offline(self, root, mode):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(QtWidgets.QLabel(
            "No internet, account, or model installation is needed for this guide."))
        self.offline_sketch = ConnectionSketch(self.instances)
        self.pages = QtWidgets.QStackedWidget()
        first = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(first)
        text = QtWidgets.QLabel(
            "1. Which unit is the root?\n\n"
            "The root is the reference sugar that MISO starts from, normally at "
            "the reducing end. For optimized orientations, donor-to-acceptor "
            "connections must lead toward it. Its position is not guessed.")
        text.setWordWrap(True)
        form.addWidget(text)
        self.root_combo = self._unit_combo()
        self.root_combo.setCurrentIndex(max(0, self.root_combo.findData(root)))
        form.addWidget(self.root_combo)
        form.addStretch()
        self.pages.addWidget(first)

        second = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(second)
        text = QtWidgets.QLabel(
            "2. Should MISO keep your rotations or optimize them?\n\n"
            "Keep: reuse the geometry and rotations you positioned.\n"
            "Optimize: reuse the same sugar geometry, but let MISO search for "
            "orientations that connect the sugars. Neither option infers chemical bonds.")
        text.setWordWrap(True)
        form.addWidget(text)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Choose orientation mode...", None)
        self.mode_combo.addItem("Keep positioned geometry and rotations", True)
        self.mode_combo.addItem("Let MISO optimize orientations", False)
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(mode)))
        form.addWidget(self.mode_combo)
        form.addStretch()
        self.pages.addWidget(second)

        third = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(third)
        text = QtWidgets.QLabel(
            "3. Which sugars are chemically connected?\n\n"
            "For each bond, choose the donor (provides the anomeric carbon), "
            "the acceptor, both carbon positions, and alpha/beta. "
            "Only hydroxyl-bearing carbons recognized by MISO are listed. "
            "Do not infer a bond just because two sugars are close together.")
        text.setWordWrap(True)
        form.addWidget(text)
        fields = QtWidgets.QFormLayout()
        self.donor_combo, self.acceptor_combo = self._unit_combo(), self._unit_combo()
        self.donor_carbon, self.acceptor_carbon = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for label, unit, carbon in (
                ("Donor:", self.donor_combo, self.donor_carbon),
                ("Acceptor:", self.acceptor_combo, self.acceptor_carbon)):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(unit, 1)
            row.addWidget(carbon)
            fields.addRow(label, row)
            unit.currentIndexChanged.connect(self._refresh_carbons)
        self.anomer_combo = QtWidgets.QComboBox()
        self.anomer_combo.addItems(["Choose...", "alpha", "beta"])
        fields.addRow("Anomer:", self.anomer_combo)
        form.addLayout(fields)
        buttons = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add this connection")
        remove = QtWidgets.QPushButton("Remove selected connection")
        add.clicked.connect(self._add_connection)
        remove.clicked.connect(self._remove_connection)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        form.addLayout(buttons)
        self.link_table = QtWidgets.QTableWidget(0, 5)
        self.link_table.setHorizontalHeaderLabels(["Donor", "Carbon", "Acceptor", "Carbon", "Anomer"])
        self.link_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.link_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.link_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        form.addWidget(self.link_table)
        self.pages.addWidget(third)
        self._show_connections()

        fourth = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(fourth)
        form.addWidget(QtWidgets.QLabel(
            "4. Review every chemical choice before applying. 'Export CSV + MISO input YAML' "
            "then writes the YAML; give it a final inspection before running MISO."))
        self.offline_review = QtWidgets.QPlainTextEdit()
        self.offline_review.setReadOnly(True)
        form.addWidget(self.offline_review)
        self.offline_approve = QtWidgets.QPushButton("Approve and apply these settings")
        self.offline_approve.clicked.connect(self._approve_offline)
        form.addWidget(self.offline_approve)
        self.pages.addWidget(fourth)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.pages)
        split.addWidget(self.offline_sketch)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        split.setSizes([550, 550])
        self.root_combo.currentIndexChanged.connect(self._refresh_offline_sketch)
        self.mode_combo.currentIndexChanged.connect(self._refresh_offline_sketch)
        self._refresh_offline_sketch()
        navigation = QtWidgets.QHBoxLayout()
        self.back_btn = QtWidgets.QPushButton("Back")
        self.back_btn.setEnabled(False)
        self.next_btn = QtWidgets.QPushButton("Next")
        self.back_btn.clicked.connect(self._back)
        self.next_btn.clicked.connect(self._next)
        navigation.addWidget(self.back_btn)
        navigation.addStretch()
        navigation.addWidget(self.next_btn)
        layout.addLayout(navigation)
        return widget

    def _refresh_carbons(self, _index=None):
        for unit, carbon in ((self.donor_combo, self.donor_carbon),
                             (self.acceptor_combo, self.acceptor_carbon)):
            carbon.clear()
            index = unit.currentData()
            if index is not None:
                carbon.addItems(linkage_carbons(self.instances[index]))

    def _show_connections(self):
        self.link_table.setRowCount(len(self.connections))
        for row, link in enumerate(self.connections):
            values = (self.instances[link.donor]["label"], link.donor_carbon,
                      self.instances[link.acceptor]["label"], link.acceptor_carbon, link.anomer)
            for column, value in enumerate(values):
                self.link_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self._refresh_offline_sketch()

    def _refresh_offline_sketch(self, _index=None):
        if hasattr(self, "root_combo") and hasattr(self, "mode_combo"):
            self.offline_sketch.set_plan(self.connections, self.root_combo.currentData(),
                                         self.mode_combo.currentData(), "Your plan")

    def _add_connection(self):
        donor, acceptor = self.donor_combo.currentData(), self.acceptor_combo.currentData()
        if donor is None or acceptor is None:
            self._error("Choose both named units first.")
            return
        link = SugarConnection(donor, self.donor_carbon.currentText(), acceptor,
                               self.acceptor_carbon.currentText(), self.anomer_combo.currentText())
        try:
            validate_connections(self.instances, self.connections + [link])
        except MISOExportError as exc:
            self._error(str(exc))
            return
        self.connections.append(link)
        self._show_connections()

    def _remove_connection(self):
        row = self.link_table.currentRow()
        if row >= 0:
            self.connections.pop(row)
            self._show_connections()

    def _offline_choices(self):
        root, mode = self.root_combo.currentData(), self.mode_combo.currentData()
        if root is None or mode is None:
            raise MISOExportError("Choose a root and an orientation mode explicitly.")
        choices = AssistantChoices(root, mode, list(self.connections))
        validate_choices(self.instances, choices)
        return choices

    def _summary(self, choices):
        lines = [
            f"Root: {self.instances[choices.root]['label']}",
            "Orientation: " + ("Keep positioned rotations" if choices.fixed_orientation
                                else "Let MISO optimize"),
            "\nConnections:",
        ]
        for link in choices.connections:
            lines.append(f"{self.instances[link.donor]['label']} {link.donor_carbon} -> "
                         f"{self.instances[link.acceptor]['label']} {link.acceptor_carbon} "
                         f"({link.anomer})")
        if not choices.connections:
            lines.append("None (single sugar).")
        return "\n".join(lines)

    def _next(self):
        page = self.pages.currentIndex()
        if page == 0 and self.root_combo.currentData() is None:
            self._error("Which named unit should be the root? Select it before continuing.")
            return
        if page == 1 and self.mode_combo.currentData() is None:
            self._error("Choose whether MISO should keep or optimize your rotations.")
            return
        if page == 2:
            try:
                self.offline_review.setPlainText(self._summary(self._offline_choices()))
            except MISOExportError as exc:
                self._error(str(exc))
                return
        self.pages.setCurrentIndex(page + 1)
        self.back_btn.setEnabled(True)
        self.next_btn.setEnabled(page + 1 < 3)

    def _back(self):
        self.pages.setCurrentIndex(max(0, self.pages.currentIndex() - 1))
        self.back_btn.setEnabled(self.pages.currentIndex() > 0)
        self.next_btn.setEnabled(True)

    def _approve_offline(self):
        try:
            self.result_choices = self._offline_choices()
        except MISOExportError as exc:
            self._error(str(exc))
            return
        self.accept()

    def _build_online(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        note = QtWidgets.QLabel(
            "Optional: needs internet, provider access, and an API key; requests may cost money. "
            "Ask an administrator to configure this once. The offline guide always works.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.service = ServiceSettingsPanel(self)
        for name in ("settings_toggle", "settings_box", "provider_combo", "endpoint_edit",
                     "model_edit", "key_edit", "auth_combo", "remember_key"):
            setattr(self, name, getattr(self.service, name))
        layout.addWidget(self.service)
        self.online_status = QtWidgets.QLabel("No molecular data is sent until you consent and press Send.")
        self.online_status.setWordWrap(True)
        layout.addWidget(self.online_status)
        self.service.status.connect(self.online_status.setText)
        self.service.changed.connect(self._reset_consent)
        self.transcript = QtWidgets.QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText(
            "Describe known chemical bonds or ask which information is missing. "
            "The model must not infer chemistry from coordinates.")
        row = QtWidgets.QHBoxLayout()
        self.message_edit = QtWidgets.QLineEdit()
        self.message_edit.setMaxLength(4000)
        self.message_edit.setPlaceholderText("Your question or molecular connectivity instructions")
        self.message_edit.returnPressed.connect(self._send)
        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.clicked.connect(self._send)
        row.addWidget(self.message_edit, 1)
        row.addWidget(self.send_btn)
        self.online_sketch = ConnectionSketch(self.instances)
        self.online_sketch.set_plan(self.connections, self.context.get("root"),
                                    self.context.get("fixed_orientation"),
                                    "Current settings (no proposal yet)")
        chat = QtWidgets.QWidget()
        chat_layout = QtWidgets.QVBoxLayout(chat)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.addWidget(self.transcript, 1)
        chat_layout.addLayout(row)
        self.online_review = QtWidgets.QPlainTextEdit()
        self.online_review.setReadOnly(True)
        self.online_review.setMaximumHeight(110)
        self.online_review.setPlaceholderText("A locally validated proposal appears here for review.")
        chat_layout.addWidget(self.online_review)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(chat)
        split.addWidget(self.online_sketch)
        split.setSizes([550, 550])
        layout.addWidget(split, 1)
        self.online_approve = QtWidgets.QPushButton("Approve and apply reviewed proposal")
        self.online_approve.setEnabled(False)
        self.online_approve.clicked.connect(self._approve_online)
        layout.addWidget(self.online_approve)
        return widget

    def _reset_consent(self):
        self._consent_config = None

    def _service_config(self):
        return self.service.service_config()

    def _load_settings(self):
        self.service.load_settings()

    def _send(self):
        if self._worker is not None:
            return
        message = self.message_edit.text().strip()
        if not message:
            self._error("Enter a question or instruction before sending.")
            return
        config = self._service_config()
        try:
            config.validate()
            if not self.key_edit.text().strip():
                raise AssistantError("Enter an API key or click Load saved key; offline guidance needs no key.")
        except AssistantError as exc:
            self.settings_toggle.setChecked(True)
            self._error(str(exc))
            return
        if self._consent_config != config:
            answer = QtWidgets.QMessageBox.question(
                self, "Send molecular information?",
                f"Send to {config.endpoint} using model '{config.model}'?\n\n"
                "This sends sugar names, labels, position indices, XYZ coordinates in Angstrom, "
                "available carbons, connection/root/mode settings, and this conversation.\n\n"
                "No images, local file paths, SMILES, or repository code are included automatically. "
                "Do not type confidential information or credentials into chat. "
                "Provider retention policies and charges apply.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                self.online_status.setText("Nothing sent. You can use the offline guide.")
                return
            self._consent_config = config
        self._online_choices = None
        self.online_approve.setEnabled(False)
        self.online_review.clear()
        self.online_sketch.set_plan(self.connections, self.context.get("root"),
                                    self.context.get("fixed_orientation"),
                                    "Current settings (waiting for proposal)")
        self._history.append({"role": "user", "content": message})
        self.transcript.appendPlainText("You: " + message)
        self.message_edit.clear()
        self.send_btn.setEnabled(False)
        self.online_status.setText("Waiting for the service (up to 45 seconds). Offline guidance remains available.")
        worker = _AdviceWorker(config, self.key_edit.text(), self.context, list(self._history))
        worker.signals.finished.connect(self._on_reply)
        self._worker = worker
        QtCore.QThreadPool.globalInstance().start(worker)

    def _on_reply(self, text, error):
        self._worker = None
        if self._closed:
            return
        self.send_btn.setEnabled(True)
        if error:
            self.online_status.setText(error)
            self._error(error)
            return
        try:
            message, choices = parse_reply(text, self.instances)
        except AssistantError as exc:
            self.online_status.setText(str(exc))
            self.transcript.appendPlainText("Local validation: " + str(exc))
            self._error(str(exc))
            return
        self._history.append({"role": "assistant", "content": text})
        self.transcript.appendPlainText("Assistant: " + message)
        self._online_choices = choices
        if choices is not None:
            self.online_review.setPlainText(self._summary(choices))
            self.online_sketch.set_plan(choices.connections, choices.root, choices.fixed_orientation,
                                        "Proposal - NOT applied until you approve")
            self.online_approve.setEnabled(True)
            self.online_status.setText("Proposal validated locally. Review every bond; nothing has been applied.")
        else:
            self.online_status.setText("Advice received. No settings changed.")

    def _approve_online(self):
        if self._online_choices is None:
            self._error("No valid proposal is available to approve.")
            return
        try:
            validate_choices(self.instances, self._online_choices)
        except MISOExportError as exc:
            self._error(str(exc))
            return
        self.result_choices = self._online_choices
        self.accept()

    def _error(self, message):
        QtWidgets.QMessageBox.warning(self, "MISO assistant", message)
