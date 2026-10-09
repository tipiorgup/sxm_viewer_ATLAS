from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

from ..._shared import QtCore, QtWidgets
from ...utils.miso_yaml import MISOExportError, build_miso_config
from ...utils.sugar_lookup import SugarLookupError, parse_sugar_description
from ._miso_connection_panel import MISOConnectionMixin
from .position_monomer_dialogs import (
    ANOMERS, ATOM_COLORS, ATOM_SIZE, RING_TYPES, _SugarLookupWorker,
    build_sugar_template, euler_to_matrix, import_monomer_engine, matrix_to_quaternion,
)


class PositionCoordinatesDialog(MISOConnectionMixin, QtWidgets.QDialog):
    """Pick XY positions and heights exactly as the MISO demo (stream.py).

    The image, pixel size and coordinate formula are taken from the demo's
    ``load_sxm_file`` + ``calculate_coordinates`` so the CSV/NPZ produced here
    are numerically identical to the demo and the notebook:

        x_ang  = orig_x * Pixelsize[0] * 10
        y_ang  = orig_y * Pixelsize[1] * 10        (orig_y flips for 'down' scans)
        Height = originalimg[orig_y, orig_x]        (Angstrom, plane-corrected)
        Z      = Height
        grid:  x = arange(W)*px*10, y = arange(H)*px*10, z = originalimg.T

    Any point may optionally get a sugar (name looked up online, or SMILES);
    built sugars are drawn on the image and exported as MISO YAML inputs that
    reference the picked points by index.
    """

    TABLE_HEADERS = ["#", "X (Å)", "Y (Å)", "Height (Å)", "Name",
                     "Sugar name / SMILES", "Ring type", "Anomer"]
    COL_NAME, COL_SUGAR, COL_RING, COL_ANOMER = 4, 5, 6, 7

    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.setWindowTitle("Position coordinates")
        self.setMinimumSize(1200, 780)

        self._instances = []             # built sugars, each with "position" = point index
        self._connections = []
        self._built_entries = None
        self._lookup_worker = None
        self._lookup_cancelled = False
        self._updating = False
        self.finished.connect(self._cancel_sugar_lookup)

        self._pick_cid = None
        self._points = []                # (orig_x, orig_y, x_ang, y_ang, height, col, row)
        self._marker_artists = []
        self._img = None                 # originalimg (Angstrom) from load_sxm_file
        self._px = (1.0, 1.0)            # Pixelsize nm/px
        self._scan_dir = ""
        self._sxm_path = None
        self._load_error = ""

        self._load_demo_image()
        self._build_ui()

    # ------------------------------------------------------------------ loading
    def _demo_loader_dir(self):
        """Directory holding the MISO demo ``sxm_loader.py``."""
        repo = Path(__file__).resolve().parents[3]
        for rel in (("MISO_demo", "app", "src"), ("MISO", "src", "src_stm")):
            cand = repo.joinpath(*rel)
            if (cand / "sxm_loader.py").exists():
                return cand
        return None

    def _resolve_sxm_path(self):
        """Find the original .sxm for the currently previewed image."""
        viewer = self.viewer
        file_key = None
        try:
            view = viewer.preview_canvas.views[0]
            file_key = view.get("path") or (view.get("meta") or {}).get("file_path")
        except Exception:
            file_key = None
        header = None
        if file_key:
            header, _ = viewer.headers.get(str(file_key), (None, None))
        # 1) explicit ConvertedSource recorded by the nanonis adapter
        if header:
            for k, v in header.items():
                if str(k).strip().lower().replace("_", "") == "convertedsource" and v:
                    p = Path(str(v))
                    if p.exists():
                        return p
        # 2) a loaded .sxm whose stem matches the previewed file
        stem = Path(str(file_key)).stem if file_key else ""
        sxm_files = [Path(f) for f in (getattr(viewer, "files", []) or [])
                     if str(f).lower().endswith(".sxm")]
        for f in sxm_files:
            if f.exists() and f.stem == stem:
                return f
        # 3) sibling <stem>.sxm next to the header
        if file_key:
            cand = Path(str(file_key)).with_suffix(".sxm")
            if cand.exists():
                return cand
        # 4) any single loaded .sxm
        for f in sxm_files:
            if f.exists():
                return f
        return None

    def _load_demo_image(self):
        """Populate self._img / self._px / self._scan_dir via load_sxm_file."""
        import numpy as np
        # Ensure the vendored nanonispy2 is importable (demo loader needs it).
        try:
            from ...providers.nanonis.adapter import _ensure_nanonis_reader
            _ensure_nanonis_reader()
        except Exception:
            pass
        sxm_path = self._resolve_sxm_path()
        if not sxm_path:
            self._load_error = "Could not locate the source .sxm file."
            return
        loader_dir = self._demo_loader_dir()
        if loader_dir is None:
            self._load_error = "MISO demo sxm_loader.py not found."
            return
        if str(loader_dir) not in sys.path:
            sys.path.insert(0, str(loader_dir))
        try:
            from sxm_loader import load_sxm_file  # demo loader
        except Exception as exc:
            self._load_error = f"Import of demo loader failed: {exc}"
            return
        try:
            data = load_sxm_file(str(sxm_path))
        except Exception as exc:
            self._load_error = f"load_sxm_file failed: {exc}"
            return
        if not data:
            self._load_error = "load_sxm_file returned no data."
            return
        try:
            self._img = np.asarray(data["originalimg"], dtype=float)
            self._px = (float(data["Pixelsize"][0]), float(data["Pixelsize"][1]))
            self._scan_dir = str(data["header"].get("scan_dir", "")).strip()
            self._sxm_path = sxm_path
        except Exception as exc:
            self._img = None
            self._load_error = f"Unexpected load_sxm_file output: {exc}"

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setSpacing(8)
        outer.setContentsMargins(12, 12, 12, 12)

        loaded = self._img is not None
        if loaded:
            info = (f"{self._sxm_path.name}  |  {self._img.shape[1]}×{self._img.shape[0]} px"
                    f"  |  {self._px[0]*10:.3f} Å/px  |  scan_dir={self._scan_dir or '?'}")
        else:
            info = f"⚠ Demo image not loaded: {self._load_error}"
        self.info_lbl = QtWidgets.QLabel(info)
        self.info_lbl.setWordWrap(True)
        outer.addWidget(self.info_lbl)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        if loaded:
            self._build_pick_canvas(left_layout)
        else:
            left_layout.addStretch()
        splitter.addWidget(left)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        right = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(right)
        layout.setSpacing(8)

        # Controls
        btn_row = QtWidgets.QHBoxLayout()
        self.pick_btn = QtWidgets.QPushButton("Pick mode: OFF")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setEnabled(loaded)
        self.pick_btn.toggled.connect(self._on_pick_toggled)
        btn_row.addWidget(self.pick_btn)
        self.clear_last_btn = QtWidgets.QPushButton("Clear last")
        self.clear_last_btn.clicked.connect(self._clear_last)
        btn_row.addWidget(self.clear_last_btn)
        self.clear_all_btn = QtWidgets.QPushButton("Clear all")
        self.clear_all_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self.clear_all_btn)
        self.count_lbl = QtWidgets.QLabel("Points: 0")
        btn_row.addWidget(self.count_lbl)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        types_row = QtWidgets.QVBoxLayout()
        types_row.addWidget(QtWidgets.QLabel("Optional monomers: Sugars"))
        self.aa_chk = QtWidgets.QCheckBox("Include amino acids (coming soon)")
        self.aa_chk.setEnabled(False)
        types_row.addWidget(self.aa_chk)
        self.lipid_chk = QtWidgets.QCheckBox("Include lipids (coming soon)")
        self.lipid_chk.setEnabled(False)
        types_row.addWidget(self.lipid_chk)
        layout.addLayout(types_row)

        hint = QtWidgets.QLabel(
            "Optional: for any point, type a sugar name (e.g. 'D chair 4C1 glucose', "
            "'chair KDO') or a SMILES, then click Build monomers. Points left blank "
            "are exported as plain coordinates.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Table
        self.table = QtWidgets.QTableWidget(0, len(self.TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(self.TABLE_HEADERS)
        header = self.table.horizontalHeader()
        for col in range(len(self.TABLE_HEADERS)):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_SUGAR, QtWidgets.QHeaderView.Interactive)
        self.table.setColumnWidth(self.COL_SUGAR, 190)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked
                                   | QtWidgets.QAbstractItemView.EditKeyPressed
                                   | QtWidgets.QAbstractItemView.AnyKeyPressed)
        self.table.setMinimumHeight(200)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.currentCellChanged.connect(self._on_point_selected)
        layout.addWidget(self.table)

        build_row = QtWidgets.QHBoxLayout()
        self.build_btn = QtWidgets.QPushButton("Build monomers")
        self.build_btn.setEnabled(loaded)
        self.build_btn.clicked.connect(self._build_monomers)
        build_row.addWidget(self.build_btn)
        self.build_status = QtWidgets.QLabel()
        self.build_status.setWordWrap(True)
        build_row.addWidget(self.build_status, 1)
        layout.addLayout(build_row)

        rot_group = QtWidgets.QGroupBox("Rotation of the selected point's sugar")
        rot_layout = QtWidgets.QHBoxLayout(rot_group)
        self.spin = {}
        for key, label in (("rx", "RX"), ("ry", "RY"), ("rz", "RZ")):
            rot_layout.addWidget(QtWidgets.QLabel(label))
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(-180.0, 180.0)
            sb.setWrapping(True)
            sb.setSingleStep(5.0)
            sb.setDecimals(1)
            sb.setSuffix(" °")
            sb.valueChanged.connect(self._on_spin_changed)
            rot_layout.addWidget(sb)
            self.spin[key] = sb
        rot_layout.addStretch()
        layout.addWidget(rot_group)

        self._build_connection_ui(layout)

        # CSV export
        csv_row = QtWidgets.QHBoxLayout()
        csv_lbl = QtWidgets.QLabel("Export CSV:")
        csv_lbl.setFixedWidth(90)
        csv_row.addWidget(csv_lbl)
        self.csv_le = QtWidgets.QLineEdit()
        self.csv_le.setPlaceholderText("circle_input.csv")
        csv_row.addWidget(self.csv_le)
        csv_browse = QtWidgets.QPushButton("Browse...")
        csv_browse.clicked.connect(self._browse_csv)
        csv_row.addWidget(csv_browse)
        layout.addLayout(csv_row)

        self.export_btn = QtWidgets.QPushButton("Export CSV (+ MISO YAML if sugars are built)")
        self.export_btn.setEnabled(loaded)
        self.export_btn.clicked.connect(self._export_csv)
        layout.addWidget(self.export_btn)
        layout.addStretch()

        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 640])
        outer.addWidget(splitter, 1)

        self._update_default_csv_name()
        self._update_rotation_controls()
        self._update_build_status()

    def _build_pick_canvas(self, layout):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self._fig = Figure(figsize=(5, 5))
        self._ax = self._fig.add_subplot(111)
        self._ax.imshow(self._img, cmap="magma", origin="lower", interpolation="nearest")
        self._ax.set_xlabel("col (px)")
        self._ax.set_ylabel("row (px)")
        self._fig.tight_layout()
        self._pick_canvas = FigureCanvas(self._fig)
        self._pick_canvas.setMinimumHeight(360)
        toolbar = NavigationToolbar(self._pick_canvas, self)
        layout.addWidget(toolbar)

        zrow = QtWidgets.QHBoxLayout()
        for label, slot in (("Reset view", self._reset_view),
                            ("Zoom out", self._zoom_out),
                            ("Zoom in", self._zoom_in)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            zrow.addWidget(b)
        zrow.addStretch()
        layout.addLayout(zrow)

        layout.addWidget(self._pick_canvas, stretch=1)

    def _reset_view(self):
        if self._img is None:
            return
        h, w = self._img.shape
        self._ax.set_xlim(-0.5, w - 0.5)
        self._ax.set_ylim(-0.5, h - 0.5)   # origin='lower'
        self._pick_canvas.draw_idle()

    def _zoom(self, factor):
        ax = getattr(self, "_ax", None)
        if ax is None:
            return
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hw, hh = (x1 - x0) / 2.0 * factor, (y1 - y0) / 2.0 * factor
        ax.set_xlim(cx - hw, cx + hw)
        ax.set_ylim(cy - hh, cy + hh)
        self._pick_canvas.draw_idle()

    def _zoom_out(self):
        self._zoom(2.0)

    def _zoom_in(self):
        self._zoom(0.5)

    # ------------------------------------------------------------------ picking
    def _on_pick_toggled(self, active):
        canvas = getattr(self, "_pick_canvas", None)
        if canvas is None:
            self.pick_btn.setChecked(False)
            return
        if active:
            self._pick_cid = canvas.mpl_connect("button_press_event", self._on_canvas_click)
            self.pick_btn.setText("Pick mode: ON")
        else:
            if self._pick_cid is not None:
                canvas.mpl_disconnect(self._pick_cid)
                self._pick_cid = None
            self.pick_btn.setText("Pick mode: OFF")

    def _on_canvas_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        if event.xdata is None or event.ydata is None or self._img is None:
            return
        H, W = self._img.shape
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        col = max(0, min(W - 1, col))
        row = max(0, min(H - 1, row))

        # Replicate the demo's calculate_coordinates exactly. The demo's click is a
        # PIL pixel on an origin='lower' render; this canvas is also origin='lower',
        # so a data-pixel click (row, col) is the PIL pixel (col, H-1-row). The demo
        # then flips the row only for 'down' scans:
        #   up   -> orig_y = H - 1 - row
        #   down -> orig_y = row + 1
        orig_x = int(col)
        orig_y = (row + 1) if self._scan_dir == "down" else (H - 1 - row)
        oy = min(max(orig_y, 0), H - 1)
        height = float(self._img[oy, orig_x])
        x_ang = orig_x * self._px[0] * 10.0
        y_ang = orig_y * self._px[1] * 10.0

        self._points.append((orig_x, orig_y, x_ang, y_ang, height, col, row))
        self._append_point_row(len(self._points) - 1)
        self._refresh_table()

    def _append_point_row(self, index):
        """Add one table row; typed sugar entries of existing rows are kept."""
        _ox, _oy, x, y, z, *_rest = self._points[index]
        self.table.blockSignals(True)
        try:
            self.table.insertRow(index)
            for c, val in enumerate([str(index), f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"]):
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.table.setItem(index, c, item)
            self.table.setItem(index, self.COL_NAME, QtWidgets.QTableWidgetItem(""))
            self.table.setItem(index, self.COL_SUGAR, QtWidgets.QTableWidgetItem(""))
        finally:
            self.table.blockSignals(False)
        for col, values in ((self.COL_RING, RING_TYPES), (self.COL_ANOMER, ANOMERS)):
            combo = QtWidgets.QComboBox()
            combo.addItems(values)
            combo.currentTextChanged.connect(self._update_build_status)
            self.table.setCellWidget(index, col, combo)

    def _refresh_table(self):
        self.count_lbl.setText(f"Points: {len(self._points)}")
        self._update_build_status()
        self._update_rotation_controls()
        self._draw_markers()

    def _clear_last(self):
        if self._points:
            self._points.pop()
            self.table.removeRow(len(self._points))
            self._refresh_table()

    def _clear_all(self):
        self._points.clear()
        self.table.setRowCount(0)
        self._refresh_table()

    # ------------------------------------------------------------------ sugars
    def _sugar_entries(self):
        """(point, name, text, ring, anomer) for every row with sugar text."""
        entries = []
        for r in range(self.table.rowCount()):
            text_item = self.table.item(r, self.COL_SUGAR)
            text = text_item.text().strip() if text_item else ""
            if not text:
                continue
            name_item = self.table.item(r, self.COL_NAME)
            name = name_item.text().strip() if name_item else ""
            ring = self.table.cellWidget(r, self.COL_RING).currentText()
            anom = self.table.cellWidget(r, self.COL_ANOMER).currentText()
            entries.append((r, name, text, ring, anom))
        return entries

    def _sugars_current(self):
        """True when built monomers still match the table entries."""
        return bool(self._instances) and self._built_entries == self._sugar_entries()

    def _on_table_item_changed(self, item):
        if item.column() in (self.COL_NAME, self.COL_SUGAR):
            self._update_build_status()

    def _update_build_status(self, *_args):
        if not hasattr(self, "build_status"):
            return
        entries = self._sugar_entries()
        if self._lookup_worker is not None:
            return
        if not entries:
            if self._instances:
                text = "Sugar entries were cleared; exports will contain coordinates only."
            else:
                text = "No monomers assigned (optional)."
        elif not self._instances or self._built_entries is None:
            text = f"{len(entries)} point(s) have sugar entries; click Build monomers."
        elif self._built_entries != entries:
            text = "Sugar entries changed since the last build; click Build monomers again."
        else:
            text = (f"Built {len(self._instances)} sugar unit(s) at point(s) "
                    + ", ".join(str(inst["position"]) for inst in self._instances) + ".")
        self.build_status.setText(text)

    @staticmethod
    def _is_smiles(text):
        from rdkit import Chem, rdBase
        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(text) is not None

    def _build_monomers(self):
        if self._lookup_worker is not None:
            return
        entries = self._sugar_entries()
        if not entries:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to build",
                "Type a sugar name or SMILES in the 'Sugar name / SMILES' column of at "
                "least one point first. Assigning monomers is optional.")
            return
        requests = []
        for r, _name, text, ring, anom in entries:
            if self._is_smiles(text):
                continue
            try:
                request = parse_sugar_description(text, ring, anom)
            except SugarLookupError as exc:
                QtWidgets.QMessageBox.warning(self, "Sugar description", f"Point {r}: {exc}")
                return
            requests.append((r, request))
        if requests:
            self._lookup_cancelled = False
            self._set_lookup_busy(True)
            self.build_status.setText(
                "Checking connection and looking up sugar names in PubChem...")
            worker = _SugarLookupWorker(requests)
            worker.signals.finished.connect(self._on_sugar_lookup_finished)
            self._lookup_worker = worker
            QtCore.QThreadPool.globalInstance().start(worker)
            return
        self._build_resolved_monomers()

    def _cancel_sugar_lookup(self, _result=None):
        self._lookup_cancelled = True

    def _set_lookup_busy(self, busy):
        loaded = self._img is not None
        self.table.setEnabled(not busy)
        self.build_btn.setEnabled(not busy and loaded)
        self.clear_last_btn.setEnabled(not busy)
        self.clear_all_btn.setEnabled(not busy)
        self.pick_btn.setEnabled(not busy and loaded)
        self.export_btn.setEnabled(not busy and loaded)

    def _on_sugar_lookup_finished(self, results, title, error):
        self._lookup_worker = None
        if self._lookup_cancelled:
            return
        self._set_lookup_busy(False)
        if error:
            if title == "No internet connection":
                error = (f"{error}\n\nNo internet connection was found, so sugar names "
                         "cannot be looked up. Enter the SMILES and details (name, ring "
                         "type, anomer) manually.")
            self.build_status.setText(title or "Sugar lookup failed.")
            QtWidgets.QMessageBox.warning(self, title or "Sugar lookup", error)
            return
        for _row, request, result in results:
            if not self._is_smiles(result.smiles):
                error = (f"PubChem returned invalid SMILES for '{request.query}'. "
                         "Enter SMILES manually.")
                self.build_status.setText(error)
                QtWidgets.QMessageBox.warning(self, "Sugar lookup", error)
                return
        self.table.blockSignals(True)
        try:
            for row, request, result in results:
                if row >= self.table.rowCount():
                    continue
                item = self.table.item(row, self.COL_SUGAR)
                item.setText(result.smiles)
                item.setToolTip(
                    f"PubChem: {request.query}\nCID: {result.cid}\n{result.iupac_name}")
                name_item = self.table.item(row, self.COL_NAME)
                if not name_item.text().strip():
                    name_item.setText(request.name)
                self.table.cellWidget(row, self.COL_RING).setCurrentText(request.ring)
                self.table.cellWidget(row, self.COL_ANOMER).setCurrentText(request.anomer)
        finally:
            self.table.blockSignals(False)
        self._build_resolved_monomers()

    def _assign_names(self, entries):
        """Fill blank names (one auto name per distinct sugar); None on a clash."""
        used = {name for _r, name, *_ in entries if name}
        auto = {}
        by_name = {}
        named = []
        for r, name, text, ring, anom in entries:
            key = (text, ring, anom)
            if not name:
                if key not in auto:
                    n = len(auto) + 1
                    while f"Sugar{n}" in used:
                        n += 1
                    auto[key] = f"Sugar{n}"
                    used.add(auto[key])
                name = auto[key]
            if by_name.setdefault(name, (r, key))[1] != key:
                QtWidgets.QMessageBox.warning(
                    self, "Sugar names",
                    f"Name '{name}' is used for different sugars (points "
                    f"{by_name[name][0]} and {r}). Points with the same Name must have the "
                    "same SMILES, ring type and anomer; give the other sugar a new name.")
                return None
            named.append((r, name, text, ring, anom))
        return named

    def _build_resolved_monomers(self):
        entries = self._assign_names(self._sugar_entries())
        if entries is None:
            return
        try:
            engine = import_monomer_engine()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "MISO engine", f"Could not import monomer builder:\n{exc}")
            return
        self.table.blockSignals(True)
        try:
            for r, name, *_ in entries:
                self.table.item(r, self.COL_NAME).setText(name)
        finally:
            self.table.blockSignals(False)

        previous = {inst["position"]: inst for inst in self._instances}
        templates = {}
        instances = []
        failures = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for r, name, text, ring, anom in entries:
                key = (text, ring, anom)
                if key not in templates:
                    templates[key] = build_sugar_template(engine, text, ring, anom)
                tmpl = templates[key]
                if tmpl is None:
                    failures.append(f"point {r}: {text} ({ring}/{anom})")
                    continue
                _ox, _oy, x_ang, y_ang, height, *_rest = self._points[r]
                old = previous.get(r)
                same = old is not None and (old["smiles"], old["ring"], old["anomer"]) == key
                instances.append({
                    "label": f"{name}.{r}", "name": name, "kind": "sugar",
                    "position": r,
                    "smiles": text, "ring": ring, "anomer": anom,
                    "conf_name": tmpl["conf_name"],
                    "rel": tmpl["rel"], "atom_types": tmpl["atom_types"],
                    "bonds": tmpl["bonds"], "rigid": tmpl.get("rigid"),
                    "functional_rel": None,
                    "com": [x_ang, y_ang, height],
                    "euler": list(old["euler"]) if same else [0.0, 0.0, 0.0],
                    "func_pos": None,
                })
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._instances = instances
        self._built_entries = self._sugar_entries()
        self._reset_connections()
        self._refresh_table()
        msg = f"Built {len(instances)} sugar unit(s) for {len(entries)} point(s) with sugar entries."
        if failures:
            msg += ("\n\nCould not build (no matching conformer):\n  "
                    + "\n  ".join(failures))
        QtWidgets.QMessageBox.information(self, "Build", msg)

    # ------------------------------------------------------------------ rotation
    def _selected_instance(self):
        row = self.table.currentRow()
        for inst in self._instances:
            if inst["position"] == row:
                return inst
        return None

    def _on_point_selected(self, *_args):
        self._update_rotation_controls()
        self._draw_markers()

    def _update_rotation_controls(self):
        if not hasattr(self, "spin"):
            return
        inst = self._selected_instance()
        self._updating = True
        try:
            for i, key in enumerate(("rx", "ry", "rz")):
                self.spin[key].setEnabled(inst is not None)
                self.spin[key].setValue(inst["euler"][i] if inst is not None else 0.0)
        finally:
            self._updating = False

    def _on_spin_changed(self, _val):
        if self._updating:
            return
        inst = self._selected_instance()
        if inst is None:
            return
        inst["euler"] = [self.spin["rx"].value(), self.spin["ry"].value(),
                         self.spin["rz"].value()]
        self._draw_markers()

    # ------------------------------------------------------------------ drawing
    def _ang_to_pixel(self, x_ang, y_ang):
        H, _W = self._img.shape
        col = x_ang / (self._px[0] * 10.0)
        orig_y = y_ang / (self._px[1] * 10.0)
        row = (orig_y - 1.0) if self._scan_dir == "down" else (H - 1.0 - orig_y)
        return col, row

    @staticmethod
    def _instance_abs_coords(inst):
        return inst["rel"] @ euler_to_matrix(*inst["euler"]).T + np.asarray(
            inst["com"], dtype=float)

    def _point_labels(self):
        names = {inst["position"]: inst["name"] for inst in self._instances}
        return [f"{i} {names[i]}" if i in names else str(i)
                for i in range(len(self._points))]

    def _draw_monomers(self, ax, active=None, artists=None):
        for inst in self._instances:
            if inst["position"] >= len(self._points):
                continue
            px = [self._ang_to_pixel(xyz[0], xyz[1])
                  for xyz in self._instance_abs_coords(inst)]
            is_active = inst is active
            for a, b in inst["bonds"]:
                ln, = ax.plot([px[a][0], px[b][0]], [px[a][1], px[b][1]], "-",
                              color="#4fc3f7" if is_active else "#90a4ae",
                              lw=1.6 if is_active else 1.0, zorder=18, alpha=0.9)
                if artists is not None:
                    artists.append(ln)
            if px:
                sc = ax.scatter([p[0] for p in px], [p[1] for p in px],
                                c=[ATOM_COLORS.get(s, "#8e24aa") for s in inst["atom_types"]],
                                s=[ATOM_SIZE.get(s, 22) for s in inst["atom_types"]],
                                edgecolors="white", linewidths=0.4, zorder=19, alpha=0.95)
                if artists is not None:
                    artists.append(sc)

    def _draw_markers(self):
        ax = getattr(self, "_ax", None)
        canvas = getattr(self, "_pick_canvas", None)
        if ax is None or canvas is None:
            return
        self._clear_markers()
        self._draw_monomers(ax, self._selected_instance(), self._marker_artists)
        for (ox, oy, x_ang, y_ang, _z, col, row), text in zip(self._points, self._point_labels()):
            dot, = ax.plot([col], [row], marker="o", color="#2196f3",
                           ms=7, mec="white", mew=0.8, zorder=20)
            lbl = ax.annotate(text, xy=(col, row), xytext=(4, 4),
                              textcoords="offset points", color="#ffee00",
                              fontsize=8, zorder=21)
            self._marker_artists.extend([dot, lbl])
        canvas.draw_idle()

    def _clear_markers(self):
        for art in getattr(self, "_marker_artists", []):
            try:
                art.remove()
            except Exception:
                pass
        self._marker_artists = []

    # ------------------------------------------------------------------ MISO
    def _assistant_blocker(self):
        if self._lookup_worker is not None:
            return "Wait for the sugar-name lookup to finish first."
        if self._instances and not self._sugars_current():
            return "Sugar entries changed since the last build; click Build monomers again."
        return super()._assistant_blocker()

    def _make_miso_config(self, out_path):
        mode = self.orientation_combo.currentData()
        if mode is None:
            raise MISOExportError(
                "Choose whether to keep or optimize the positioned orientations.")
        return build_miso_config(
            self._instances, self._connections, self.root_combo.currentData(),
            mode, out_path, self._sxm_path, positions_csv=out_path)

    # ------------------------------------------------------------------ export
    def _browse_csv(self):
        default_name = self.csv_le.text().strip() or "circle_input.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", default_name, "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.csv_le.setText(path)

    def _export_csv(self):
        import csv
        if self._img is None:
            QtWidgets.QMessageBox.warning(self, "No image", self._load_error or "Image not loaded.")
            return
        if not self._points:
            QtWidgets.QMessageBox.warning(self, "No points", "Add at least one point first.")
            return
        if self._lookup_worker is not None:
            QtWidgets.QMessageBox.warning(
                self, "Sugar lookup", "Wait for the sugar-name lookup to finish first.")
            return
        out_path = self.csv_le.text().strip() or "circle_input.csv"
        use_sugars = self._sugars_current()
        if self._sugar_entries() and not use_sugars:
            answer = QtWidgets.QMessageBox.question(
                self, "Monomers not built",
                "Some points have sugar entries that are not built (or changed since the "
                "last build), so no MISO YAML can be written.\n\n"
                "Export coordinates only? Choose No to go back and click Build monomers.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        config = None
        export_errors = (OSError,)
        if use_sugars:
            try:
                config = self._make_miso_config(out_path)
            except MISOExportError as exc:
                QtWidgets.QMessageBox.warning(self, "MISO YAML export", str(exc))
                return
            try:
                import yaml
            except ImportError:
                QtWidgets.QMessageBox.critical(
                    self, "Missing dependency", "PyYAML is required for MISO YAML export.")
                return
            export_errors += (yaml.YAMLError,)
        yaml_path = Path(out_path).with_suffix(".yml")
        try:
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Point", "Original_X", "Original_Y",
                                 "X (Angstrom)", "Y (Angstrom)", "Height", "Z (Angstrom)"])
                for i, (ox, oy, x, y, z, *_rest) in enumerate(self._points):
                    # Height in Angstrom from originalimg; Z defaults to 0.
                    writer.writerow([i, ox, oy, x, y, z, 0.0])
            self._export_npz(out_path)
            self._export_png(out_path)
            extra = []
            if config is not None:
                extra.append(self._export_monomer_pickle(out_path))
                extra.append(self._export_orientations(out_path))
                with open(yaml_path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        except export_errors as exc:
            QtWidgets.QMessageBox.critical(
                self, "Export failed",
                f"Could not complete the export:\n{exc}\n\n"
                "Some files may already have been written. Fix the error and export again.")
            return
        message = (f"Saved {len(self._points)} points to:\n{out_path}\n"
                   f"NPZ: {Path(out_path).with_suffix('.npz').name}\n"
                   f"PNG: {Path(out_path).with_suffix('.png').name}")
        if config is not None:
            self.viewer.last_monomer_yaml = str(yaml_path.resolve())
            message += ("\n" + "\n".join(Path(p).name for p in extra)
                        + f"\n\nMISO input YAML ({len(self._instances)} sugar unit(s)):\n  "
                        + str(yaml_path.resolve())
                        + "\n\nPlease give the YAML a final inspection (units, root, "
                          "linkages, \u03b1/\u03b2, orientation mode) before running MISO. "
                          "Open Run MISO to use this YAML and its companion files.")
        QtWidgets.QMessageBox.information(self, "Done", message)

    def _export_monomer_pickle(self, out_path):
        """{name: rigid} so MISO reuses the exact built geometry."""
        import pickle
        monomer_data = {}
        for inst in self._instances:
            if inst.get("rigid") is not None:
                monomer_data.setdefault(inst["name"], inst["rigid"])
        pkl_path = f"{Path(out_path).with_suffix('')}_monomer_data.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(monomer_data, f)
        return pkl_path

    def _export_orientations(self, out_path):
        """Orientation CSV keyed by the point index used in the YAML."""
        import csv
        ori_path = f"{Path(out_path).with_suffix('')}_orientations.csv"
        with open(ori_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Point", "Instance", "Type", "Conformer",
                        "quat_x", "quat_y", "quat_z", "quat_w",
                        "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"])
            for inst in self._instances:
                Rm = euler_to_matrix(*inst["euler"])
                q = matrix_to_quaternion(Rm)
                row = [inst["position"], inst["label"], "sugar", inst["conf_name"],
                       f"{q[0]:.8f}", f"{q[1]:.8f}", f"{q[2]:.8f}", f"{q[3]:.8f}"]
                row += [f"{v:.8f}" for v in Rm.flatten()]
                w.writerow(row)
        return ori_path

    def _export_npz(self, csv_path):
        if self._img is None:
            return
        H, W = self._img.shape
        x_ang = np.arange(W) * (self._px[0] * 10.0)
        y_ang = np.arange(H) * (self._px[1] * 10.0)
        z_grid = self._img.T                       # demo convention: originalimg.T
        npz_path = Path(csv_path).with_suffix(".npz")
        np.savez(str(npz_path), x=x_ang, y=y_ang, z=z_grid)

    def _export_png(self, csv_path):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        if self._img is None:
            return
        fig = Figure(figsize=(6, 6), dpi=150)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.imshow(self._img, cmap="magma", origin="lower", interpolation="nearest")
        self._draw_monomers(ax)
        for (ox, oy, x_ang, y_ang, _z, col, row), text in zip(self._points, self._point_labels()):
            ax.plot(col, row, marker="o", color="#2196f3", ms=6, mec="white", mew=0.7, zorder=20)
            ax.annotate(text, xy=(col, row), xytext=(4, 4),
                        textcoords="offset points", color="#ffee00", fontsize=7, zorder=21)
        ax.set_xlabel("col (px)")
        ax.set_ylabel("row (px)")
        ax.set_title(Path(csv_path).stem)
        fig.tight_layout()
        fig.savefig(str(Path(csv_path).with_suffix(".png")), dpi=150)

    def _update_default_csv_name(self):
        stem = ""
        try:
            if self._sxm_path is not None:
                stem = self._sxm_path.stem
            elif self.viewer.preview_canvas.views:
                stem = Path(self.viewer.preview_canvas.views[0].get("file_name", "")).stem
        except Exception:
            pass
        if stem:
            self.csv_le.setText(f"{stem}_positions.csv")

    def closeEvent(self, event):
        self._cancel_sugar_lookup()
        canvas = getattr(self, "_pick_canvas", None)
        if self._pick_cid is not None and canvas is not None:
            canvas.mpl_disconnect(self._pick_cid)
        super().closeEvent(event)
