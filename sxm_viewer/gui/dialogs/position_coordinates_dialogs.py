from __future__ import annotations
from ..._shared import QtCore, QtWidgets


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
        from ..thumbnail_render import sample_array_value
        z_raw = sample_array_value(view.get("arr"), x_nm, y_nm, view.get("extent"))
        z_ang = (z_raw * 10.0) if z_raw is not None else 0.0
        self._points.append((x_nm * 10.0, y_nm * 10.0, z_ang))
        self._refresh_table()

    def _refresh_table(self):
        self.table.setRowCount(len(self._points))
        for i, (x, y, z) in enumerate(self._points):
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
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", "circle_input.csv", "CSV Files (*.csv);;All Files (*)"
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
            writer.writerow(["X (Angstrom)", "Y (Angstrom)", "Height"])
            for x, y, z in self._points:
                writer.writerow([x, y, z])
        QtWidgets.QMessageBox.information(self, "Done", f"Saved {len(self._points)} points to:\n{out_path}")

    def _draw_markers(self):
        canvas = getattr(self.viewer, "preview_canvas", None)
        if canvas is None or canvas.main_ax is None:
            return
        self._clear_markers(canvas)
        for i, (x_ang, y_ang, _) in enumerate(self._points):
            x_nm = x_ang / 10.0
            y_nm = y_ang / 10.0
            dot, = canvas.main_ax.plot(
                [x_nm], [y_nm], marker='o', color='#ff5252',
                ms=7, mec='white', mew=0.8, zorder=20
            )
            lbl = canvas.main_ax.annotate(
                str(i), xy=(x_nm, y_nm),
                xytext=(4, 4), textcoords='offset points',
                color='white', fontsize=7, zorder=21
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