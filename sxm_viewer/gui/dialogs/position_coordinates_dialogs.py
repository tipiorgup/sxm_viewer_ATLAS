from __future__ import annotations
from pathlib import Path
from ..._shared import QtCore, QtWidgets
from ..thumbnail_render import sample_array_value


class PositionCoordinatesDialog(QtWidgets.QDialog):

    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.setWindowTitle("Position coordinates")
        self.setMinimumWidth(500)
        self._build_ui()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        self._pick_cid = None
        self._points = []
        self._marker_artists = []

        # Pick mode toggle
        btn_row = QtWidgets.QHBoxLayout()
        self.pick_btn = QtWidgets.QPushButton("Pick mode: OFF")
        self.pick_btn.setCheckable(True)
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
        self.table.setFixedHeight(200)
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
        export_btn.clicked.connect(self._export_csv)
        layout.addWidget(export_btn)
        self._update_default_csv_name()

    def _on_pick_toggled(self, active):
        canvas = getattr(self.viewer, "preview_canvas", None)
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
        canvas = getattr(self.viewer, "preview_canvas", None)
        if canvas is None or not canvas.views:
            return
        view = canvas.views[0]
        x_nm = event.xdata
        y_nm = event.ydata
        if x_nm is None or y_nm is None:
            return

        extent = view.get("extent")
        arr = view.get("arr")
        if arr is None or extent is None:
            return

        h, w = arr.shape
        xmin, xmax, ymin, ymax = extent
        pixel_x = int((x_nm - xmin) / (xmax - xmin) * w)
        pixel_y = int((y_nm - ymin) / (ymax - ymin) * h)
        pixel_x = max(0, min(w - 1, pixel_x))
        pixel_y = max(0, min(h - 1, pixel_y))

        # Pixel-based Angstrom coords, matching NPZ grid (origin at 0)
        scan_width_ang = abs(xmax - xmin) * 10.0   # nm → Å
        scan_height_ang = abs(ymax - ymin) * 10.0
        x_ang = pixel_x / w * scan_width_ang
        y_ang = pixel_y / h * scan_height_ang

        import numpy as np
        z_raw = sample_array_value(arr, x_nm, y_nm, extent)
        if z_raw is not None:
            display_unit = (view.get("unit") or "nm").strip()
            unit_to_angstrom = {
                "pm": 0.01,
                "nm": 10.0,
                "m": 1e10,
                "Angstrom": 1.0,
                "A": 1.0,
            }
            factor = unit_to_angstrom.get(display_unit, 10.0)
            z_ang = (z_raw - float(np.nanmin(arr))) * factor
        else:
            z_ang = 0.0

        self._points.append((pixel_x, pixel_y, x_ang, y_ang, z_ang, x_nm, y_nm))
        self._refresh_table()

    def _refresh_table(self):
        self.table.setRowCount(len(self._points))
        for i, (px, py, x, y, z, *_) in enumerate(self._points):
            for col, val in enumerate([str(i), f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"]):
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(i, col, item)
        self.count_lbl.setText(f"Points: {len(self._points)}")
        self._draw_markers()

    def _clear_last(self):
        if self._points:
            self._points.pop()
            self._refresh_table()

    def _clear_all(self):
        self._points.clear()
        self._refresh_table()


    def _browse_csv(self):
        default_name = self.csv_le.text().strip() or "circle_input.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", default_name, "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.csv_le.setText(path)

    def _export_csv(self):
        import csv
        if not self._points:
            QtWidgets.QMessageBox.warning(self, "No points", "Add at least one point first.")
            return
        out_path = self.csv_le.text().strip() or "circle_input.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Point", "Original_X", "Original_Y", "X (Angstrom)", "Y (Angstrom)", "Height", "Z (Angstrom)"])
            for i, (px, py, x, y, z, *_) in enumerate(self._points):
                writer.writerow([i, px, py, x, y, z, 0])
        self._export_npz(out_path)
        self._export_png(out_path)
        QtWidgets.QMessageBox.information(
            self, "Done",
            f"Saved {len(self._points)} points to:\n{out_path}\n"
            f"NPZ: {Path(out_path).with_suffix('.npz').name}\n"
            f"PNG: {Path(out_path).with_suffix('.png').name}"
        )

    def _draw_markers(self):
        canvas = getattr(self.viewer, "preview_canvas", None)
        if canvas is None or canvas.main_ax is None:
            return
        self._clear_markers(canvas)
        for i, (px, py, x_ang, y_ang, _z, x_nm, y_nm) in enumerate(self._points):
            dot, = canvas.main_ax.plot(
                [x_nm], [y_nm], marker='o', color='#ff5252',
                ms=7, mec='white', mew=0.8, zorder=20
            )
            lbl = canvas.main_ax.annotate(
                str(i), xy=(x_nm, y_nm),
                xytext=(4, 4), textcoords='offset points',
                color='#ffee00', fontsize=7, zorder=21
            )
            self._marker_artists.append((dot, lbl))
        canvas.draw_idle()

    def _clear_markers(self, canvas=None):
        if canvas is None:
            canvas = getattr(self.viewer, "preview_canvas", None)
        for dot, lbl in getattr(self, "_marker_artists", []):
            try:
                dot.remove()
                lbl.remove()
            except Exception:
                pass
        self._marker_artists = []
        if canvas is not None:
            canvas.draw_idle()

    def closeEvent(self, event):
        if self._pick_cid is not None:
            canvas = getattr(self.viewer, "preview_canvas", None)
            if canvas is not None:
                canvas.mpl_disconnect(self._pick_cid)
        self._clear_markers()
        super().closeEvent(event)

    def _update_default_csv_name(self):
        stem = ""
        try:
            canvas = getattr(self.viewer, "preview_canvas", None)
            if canvas and canvas.views:
                stem = Path(canvas.views[0].get("file_name", "")).stem
            if not stem:
                stem = Path(self.viewer.last_preview[0]).stem
        except Exception:
            pass
        if stem:
            self.csv_le.setText(f"{stem}_positions.csv")

    def _export_npz(self, csv_path):
        import numpy as np
        canvas = getattr(self.viewer, "preview_canvas", None)
        if canvas is None or not canvas.views:
            return
        view = canvas.views[0]
        arr = view.get("arr")
        extent = view.get("extent")
        if arr is None or extent is None:
            return

        display_unit = (view.get("unit") or "nm").strip()
        unit_to_angstrom = {
            "pm": 0.01,
            "nm": 10.0,
            "m": 1e10,
            "Angstrom": 1.0,
            "A": 1.0,
        }
        factor = unit_to_angstrom.get(display_unit, 10.0)

        h, w = arr.shape
        x0, x1, y_top, y_bot = extent
        scan_x_ang = abs(x1 - x0) * 10.0   # nm → Å
        scan_y_ang = abs(y_top - y_bot) * 10.0

        # Pixel-size axes starting at 0, matching the reference sxm_data convention
        x_ang = np.arange(w) * (scan_x_ang / w)
        y_ang = np.arange(h) * (scan_y_ang / h)

        # Corrugation: baseline-subtract then convert to Å; fill NaN with 0
        z = (arr - np.nanmin(arr)) * factor
        z = np.nan_to_num(z, nan=0.0)

        # ATLAS expects z.shape = (W, H): z[i,j] = height at x[i], y[j]
        z_grid = z.T

        npz_path = Path(csv_path).with_suffix(".npz")
        np.savez(str(npz_path), x=x_ang, y=y_ang, z=z_grid)

    def _export_png(self, csv_path):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")  # off-screen backend for the export figure
        import matplotlib.pyplot as plt

        canvas = getattr(self.viewer, "preview_canvas", None)
        if canvas is None or not canvas.views:
            return
        view = canvas.views[0]
        arr = view.get("arr")
        if arr is None:
            return

        # Match exactly what the viewer rendered: read origin + display extent
        # from the canvas metadata so we don't guess.
        meta = {}
        if canvas.main_ax is not None:
            meta = getattr(canvas, "_image_meta", {}).get(canvas.main_ax, {})
        origin = meta.get("origin", "upper")
        extent = meta.get("extent") or view.get("extent")

        arr_plot = np.flipud(arr) if origin == "lower" else np.asarray(arr)
        cmap = view.get("cmap", "gray")

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        imshow_kw = dict(origin=origin, interpolation="nearest", cmap=cmap, aspect="equal")
        if extent is not None:
            ax.imshow(arr_plot, extent=extent, **imshow_kw)
        else:
            ax.imshow(arr_plot, **imshow_kw)

        for i, (px, py, x_ang, y_ang, _z, x_nm, y_nm) in enumerate(self._points):
            ax.plot(x_nm, y_nm, marker='o', color='#ff5252',
                    ms=6, mec='white', mew=0.7, zorder=20)
            ax.annotate(str(i), xy=(x_nm, y_nm),
                        xytext=(4, 4), textcoords='offset points',
                        color='#ffee00', fontsize=7, zorder=21)

        ax.set_xlabel(f"x ({view.get('unit', 'nm')})")
        ax.set_ylabel(f"y ({view.get('unit', 'nm')})")
        ax.set_title(Path(csv_path).stem)
        fig.tight_layout()

        png_path = Path(csv_path).with_suffix(".png")
        fig.savefig(str(png_path), dpi=150)
        plt.close(fig)
