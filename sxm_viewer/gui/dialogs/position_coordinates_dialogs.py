from __future__ import annotations
import sys
from pathlib import Path
from ..._shared import QtCore, QtWidgets


class PositionCoordinatesDialog(QtWidgets.QDialog):
    """Pick XY positions and heights exactly as the MISO demo (stream.py).

    The image, pixel size and coordinate formula are taken from the demo's
    ``load_sxm_file`` + ``calculate_coordinates`` so the CSV/NPZ produced here
    are numerically identical to the demo and the notebook:

        x_ang  = orig_x * Pixelsize[0] * 10
        y_ang  = orig_y * Pixelsize[1] * 10        (orig_y flips for 'down' scans)
        Height = originalimg[orig_y, orig_x]        (Angstrom, plane-corrected)
        Z      = Height
        grid:  x = arange(W)*px*10, y = arange(H)*px*10, z = originalimg.T
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.setWindowTitle("Position coordinates")
        self.setMinimumSize(620, 720)

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
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        loaded = self._img is not None
        if loaded:
            info = (f"{self._sxm_path.name}  |  {self._img.shape[1]}×{self._img.shape[0]} px"
                    f"  |  {self._px[0]*10:.3f} Å/px  |  scan_dir={self._scan_dir or '?'}")
        else:
            info = f"⚠ Demo image not loaded: {self._load_error}"
        self.info_lbl = QtWidgets.QLabel(info)
        self.info_lbl.setWordWrap(True)
        layout.addWidget(self.info_lbl)

        if loaded:
            self._build_pick_canvas(layout)

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

        # Table
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "X (Å)", "Y (Å)", "Height (Å)"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setFixedHeight(170)
        layout.addWidget(self.table)

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

        export_btn = QtWidgets.QPushButton("Export CSV")
        export_btn.setEnabled(loaded)
        export_btn.clicked.connect(self._export_csv)
        layout.addWidget(export_btn)
        self._update_default_csv_name()

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
        self._refresh_table()

    def _refresh_table(self):
        self.table.setRowCount(len(self._points))
        for i, (ox, oy, x, y, z, *_rest) in enumerate(self._points):
            for c, val in enumerate([str(i), f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"]):
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(i, c, item)
        self.count_lbl.setText(f"Points: {len(self._points)}")
        self._draw_markers()

    def _clear_last(self):
        if self._points:
            self._points.pop()
            self._refresh_table()

    def _clear_all(self):
        self._points.clear()
        self._refresh_table()

    def _draw_markers(self):
        ax = getattr(self, "_ax", None)
        canvas = getattr(self, "_pick_canvas", None)
        if ax is None or canvas is None:
            return
        self._clear_markers()
        for i, (ox, oy, x_ang, y_ang, _z, col, row) in enumerate(self._points):
            dot, = ax.plot([col], [row], marker="o", color="#2196f3",
                           ms=7, mec="white", mew=0.8, zorder=20)
            lbl = ax.annotate(str(i), xy=(col, row), xytext=(4, 4),
                              textcoords="offset points", color="#ffee00",
                              fontsize=8, zorder=21)
            self._marker_artists.append((dot, lbl))
        canvas.draw_idle()

    def _clear_markers(self):
        for dot, lbl in getattr(self, "_marker_artists", []):
            try:
                dot.remove()
                lbl.remove()
            except Exception:
                pass
        self._marker_artists = []

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
        out_path = self.csv_le.text().strip() or "circle_input.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Point", "Original_X", "Original_Y",
                             "X (Angstrom)", "Y (Angstrom)", "Height", "Z (Angstrom)"])
            for i, (ox, oy, x, y, z, *_rest) in enumerate(self._points):
                # Height in Angstrom from originalimg; Z defaults to 0.
                writer.writerow([i, ox, oy, x, y, z, 0.0])
        self._export_npz(out_path)
        self._export_png(out_path)
        QtWidgets.QMessageBox.information(
            self, "Done",
            f"Saved {len(self._points)} points to:\n{out_path}\n"
            f"NPZ: {Path(out_path).with_suffix('.npz').name}\n"
            f"PNG: {Path(out_path).with_suffix('.png').name}"
        )

    def _export_npz(self, csv_path):
        import numpy as np
        if self._img is None:
            return
        H, W = self._img.shape
        x_ang = np.arange(W) * (self._px[0] * 10.0)
        y_ang = np.arange(H) * (self._px[1] * 10.0)
        z_grid = self._img.T                       # demo convention: originalimg.T
        npz_path = Path(csv_path).with_suffix(".npz")
        np.savez(str(npz_path), x=x_ang, y=y_ang, z=z_grid)

    def _export_png(self, csv_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if self._img is None:
            return
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(self._img, cmap="magma", origin="lower", interpolation="nearest")
        for i, (ox, oy, x_ang, y_ang, _z, col, row) in enumerate(self._points):
            ax.plot(col, row, marker="o", color="#2196f3", ms=6, mec="white", mew=0.7, zorder=20)
            ax.annotate(str(i), xy=(col, row), xytext=(4, 4),
                        textcoords="offset points", color="#ffee00", fontsize=7, zorder=21)
        ax.set_xlabel("col (px)")
        ax.set_ylabel("row (px)")
        ax.set_title(Path(csv_path).stem)
        fig.tight_layout()
        fig.savefig(str(Path(csv_path).with_suffix(".png")), dpi=150)
        plt.close(fig)

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
        canvas = getattr(self, "_pick_canvas", None)
        if self._pick_cid is not None and canvas is not None:
            canvas.mpl_disconnect(self._pick_cid)
        super().closeEvent(event)
