from __future__ import annotations
import shutil
from datetime import datetime
from pathlib import Path

from ..._shared import QtCore, QtWidgets, QtGui
from ..canvases.molecular_overlay import Molecule


_AVOGADRO_CANDIDATES = [
    "avogadro2",
    r"C:\Program Files\Avogadro2\bin\avogadro2.exe",
    r"C:\Program Files (x86)\Avogadro2\bin\avogadro2.exe",
    str(Path.home() / "AppData" / "Local" / "Programs" / "Avogadro2" / "bin" / "avogadro2.exe"),
    "/usr/bin/avogadro2",
    "/usr/local/bin/avogadro2",
    "/Applications/Avogadro2.app/Contents/MacOS/Avogadro2",
]


def _detect_avogadro() -> str:
    for candidate in _AVOGADRO_CANDIDATES:
        found = shutil.which(candidate) or (Path(candidate).is_file() and candidate)
        if found:
            return str(found)
    return ""


class AvogadroDialog(QtWidgets.QDialog):
    """Launch Avogadro for interactive molecule editing and hot-reload the overlay."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.setWindowTitle("Avogadro Integration")
        self.setMinimumWidth(560)
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._watched_path: str = ""
        self._build_ui()
        self._prefill()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        # Molecule file
        self.mol_le = QtWidgets.QLineEdit()
        self.mol_le.setPlaceholderText("path/to/molecule.mol")
        self.mol_le.textChanged.connect(self._on_mol_path_changed)
        mol_browse = QtWidgets.QPushButton("Browse…")
        mol_browse.setFixedWidth(70)
        mol_browse.clicked.connect(self._browse_mol)
        mol_row = QtWidgets.QHBoxLayout()
        mol_row.addWidget(self.mol_le)
        mol_row.addWidget(mol_browse)
        form.addRow("Molecule file:", mol_row)

        # Avogadro executable
        self.avo_le = QtWidgets.QLineEdit()
        self.avo_le.setPlaceholderText("avogadro2  (auto-detected)")
        avo_browse = QtWidgets.QPushButton("Browse…")
        avo_browse.setFixedWidth(70)
        avo_browse.clicked.connect(self._browse_avo)
        avo_detect = QtWidgets.QPushButton("Auto-detect")
        avo_detect.setFixedWidth(85)
        avo_detect.clicked.connect(self._auto_detect)
        avo_row = QtWidgets.QHBoxLayout()
        avo_row.addWidget(self.avo_le)
        avo_row.addWidget(avo_browse)
        avo_row.addWidget(avo_detect)
        form.addRow("Avogadro:", avo_row)
        root.addLayout(form)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("Open in Avogadro")
        self.open_btn.clicked.connect(self._open_in_avogadro)
        self.reload_btn = QtWidgets.QPushButton("Reload in viewer")
        self.reload_btn.clicked.connect(self._reload_in_viewer)
        self.load_btn = QtWidgets.QPushButton("Load as overlay")
        self.load_btn.clicked.connect(self._load_as_overlay)
        self.watch_chk = QtWidgets.QCheckBox("Auto-reload on save")
        self.watch_chk.toggled.connect(self._on_watch_toggled)
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.reload_btn)
        btn_row.addWidget(self.load_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.watch_chk)
        root.addLayout(btn_row)

        # Status
        self.status_lbl = QtWidgets.QLabel("Ready.")
        self.status_lbl.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(self.status_lbl)

    # ── Pre-fill ──────────────────────────────────────────────────────────────

    def _prefill(self):
        # Avogadro path
        detected = _detect_avogadro()
        if detected:
            self.avo_le.setText(detected)
            self.avo_le.setPlaceholderText("")

        # Molecule from canvas
        canvas = getattr(self.viewer, "preview_canvas", None)
        if canvas and getattr(canvas, "molecules", None):
            mol = canvas.molecules[-1]
            if mol.filepath:
                self.mol_le.setText(str(mol.filepath))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _browse_mol(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open molecule", "",
            "Molecule files (*.mol *.mol2 *.sdf *.xyz *.pdb);;All files (*)"
        )
        if path:
            self.mol_le.setText(path)

    def _browse_avo(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Avogadro executable", "", "Executables (*.exe);;All files (*)"
        )
        if path:
            self.avo_le.setText(path)

    def _auto_detect(self):
        found = _detect_avogadro()
        if found:
            self.avo_le.setText(found)
            self._set_status(f"Found: {found}")
        else:
            self._set_status("Avogadro not found — install it or browse manually.")

    def _set_status(self, msg: str):
        self.status_lbl.setText(msg)

    def _mol_path(self) -> str:
        return self.mol_le.text().strip()

    def _avo_exe(self) -> str:
        return self.avo_le.text().strip() or "avogadro2"

    # ── Watch ─────────────────────────────────────────────────────────────────

    def _on_mol_path_changed(self, text: str):
        path = text.strip()
        if self._watched_path and self._watched_path != path:
            self._watcher.removePath(self._watched_path)
            self._watched_path = ""
        if self.watch_chk.isChecked() and path and Path(path).is_file():
            self._start_watch(path)

    def _on_watch_toggled(self, checked: bool):
        path = self._mol_path()
        if checked and path and Path(path).is_file():
            self._start_watch(path)
        elif not checked and self._watched_path:
            self._watcher.removePath(self._watched_path)
            self._watched_path = ""

    def _start_watch(self, path: str):
        if self._watched_path:
            self._watcher.removePath(self._watched_path)
        self._watcher.addPath(path)
        self._watched_path = path

    def _on_file_changed(self, path: str):
        # Re-add path because some editors replace-then-rename (QFileSystemWatcher loses track)
        QtCore.QTimer.singleShot(300, lambda: self._delayed_reload(path))

    def _delayed_reload(self, path: str):
        if Path(path).is_file():
            self._watcher.addPath(path)
            self._reload_molecule(path)
            ts = datetime.now().strftime("%H:%M:%S")
            self._set_status(f"Auto-reloaded at {ts}")

    # ── Actions ───────────────────────────────────────────────────────────────

    def _open_in_avogadro(self):
        path = self._mol_path()
        if not path:
            QtWidgets.QMessageBox.warning(self, "No file", "Select a molecule file first.")
            return
        if not Path(path).is_file():
            QtWidgets.QMessageBox.warning(self, "Not found", f"File not found:\n{path}")
            return
        exe = self._avo_exe()
        ok = QtCore.QProcess.startDetached(exe, [path])
        if ok:
            self._set_status(f"Launched Avogadro with {Path(path).name}")
        else:
            QtWidgets.QMessageBox.critical(
                self, "Launch failed",
                f"Could not start Avogadro.\nExecutable: {exe}\n\n"
                "Make sure Avogadro2 is installed and the path is correct."
            )

    def _load_as_overlay(self):
        path = self._mol_path()
        if not path or not Path(path).is_file():
            QtWidgets.QMessageBox.warning(self, "No file", "Select a valid molecule file first.")
            return
        canvas = getattr(self.viewer, "preview_canvas", None)
        if canvas is None:
            return
        canvas.add_molecule(path)
        self.viewer.on_show_molecules_toggled(True)
        self._set_status(f"Loaded {Path(path).name} as overlay.")

    def _reload_in_viewer(self):
        path = self._mol_path()
        if not path or not Path(path).is_file():
            QtWidgets.QMessageBox.warning(self, "No file", "Select a valid molecule file first.")
            return
        self._reload_molecule(path)
        self._set_status(f"Reloaded {Path(path).name} at {datetime.now().strftime('%H:%M:%S')}")

    def _reload_molecule(self, path: str):
        canvas = getattr(self.viewer, "preview_canvas", None)
        if canvas is None:
            return

        norm = str(Path(path).resolve())
        existing = [
            m for m in getattr(canvas, "molecules", [])
            if m.filepath and str(Path(m.filepath).resolve()) == norm
        ]

        if existing:
            # Replace in-place, preserving position/orientation
            old = existing[-1]
            old_offset = old.offset.copy()
            old_angles = old.angles.copy()
            old_scale  = old.scale
            old_mirror_x = old.mirror_x
            old_mirror_y = old.mirror_y

            canvas._push_molecule_snapshot()
            canvas.molecules.remove(old)

            new_mol = Molecule(path)
            new_mol.offset   = old_offset
            new_mol.angles   = old_angles
            new_mol.scale    = old_scale
            new_mol.mirror_x = old_mirror_x
            new_mol.mirror_y = old_mirror_y
            canvas.molecules.append(new_mol)
            canvas._active_molecule_idx = len(canvas.molecules) - 1
        else:
            canvas.add_molecule(path)

        self.viewer.on_show_molecules_toggled(True)
        try:
            canvas._redraw()
        except Exception:
            canvas.draw_idle()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._watched_path:
            self._watcher.removePath(self._watched_path)
        super().closeEvent(event)
