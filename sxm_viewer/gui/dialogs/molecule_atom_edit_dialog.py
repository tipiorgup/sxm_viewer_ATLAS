"""In-canvas interactive atom editor for molecule overlays."""
from __future__ import annotations
import tempfile
import numpy as np
from pathlib import Path

from ..._shared import QtCore, QtWidgets, QtGui
from ..canvases.molecular_overlay import Molecule, get_atom_color


def _inv_transform_matrix(mol: Molecule) -> np.ndarray:
    """Return M (3x3) so that local_delta = M @ canvas_delta_nm."""
    rads = np.radians(mol.angles)
    cx, cy, cz = np.cos(rads)
    sx, sy, sz = np.sin(rads)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx,  cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0,  cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    D = np.diag([-1.0 if mol.mirror_x else 1.0,
                 -1.0 if mol.mirror_y else 1.0,
                 1.0])
    return (R.T @ D) / mol.scale


class MoleculeAtomEditDialog(QtWidgets.QDialog):
    """Drag atoms directly on the STM canvas to reposition them."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.setWindowTitle("Edit Atoms")
        self.setWindowFlags(QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(540)

        self._edit_active = False
        self._selected_idx = -1
        self._press_canvas_xy: tuple | None = None
        self._press_atom_coords: np.ndarray | None = None
        self._press_cid = None
        self._motion_cid = None
        self._release_cid = None
        self._handle_artists: list = []

        self._build_ui()
        self._populate_molecules()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(QtWidgets.QLabel("Molecule:"))
        self.mol_combo = QtWidgets.QComboBox()
        self.mol_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sel_row.addWidget(self.mol_combo)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setFixedWidth(70)
        self.refresh_btn.clicked.connect(self._populate_molecules)
        sel_row.addWidget(self.refresh_btn)
        root.addLayout(sel_row)

        btn_row = QtWidgets.QHBoxLayout()
        self.edit_btn = QtWidgets.QPushButton("Start editing")
        self.edit_btn.setCheckable(True)
        self.edit_btn.toggled.connect(self._on_edit_toggled)
        btn_row.addWidget(self.edit_btn)

        self.relax_btn = QtWidgets.QPushButton("Relax (MMFF)")
        self.relax_btn.clicked.connect(self._relax_mmff)
        btn_row.addWidget(self.relax_btn)

        self.rebond_btn = QtWidgets.QPushButton("Recalc bonds")
        self.rebond_btn.clicked.connect(self._recalc_bonds)
        btn_row.addWidget(self.rebond_btn)

        self.save_btn = QtWidgets.QPushButton("Save to file")
        self.save_btn.clicked.connect(self._save_to_file)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self.info_lbl = QtWidgets.QLabel("Select a molecule and press 'Start editing'.")
        self.info_lbl.setStyleSheet("color: #888; font-size: 11px;")
        self.info_lbl.setWordWrap(True)
        root.addWidget(self.info_lbl)

    # ── Molecule helpers ──────────────────────────────────────────────────

    def _canvas(self):
        return getattr(self.viewer, "preview_canvas", None)

    def _molecules(self) -> list:
        c = self._canvas()
        return list(getattr(c, "molecules", [])) if c else []

    def _active_mol(self) -> Molecule | None:
        idx = self.mol_combo.currentIndex()
        mols = self._molecules()
        return mols[idx] if 0 <= idx < len(mols) else None

    def _populate_molecules(self):
        mols = self._molecules()
        prev = self.mol_combo.currentIndex()
        self.mol_combo.blockSignals(True)
        self.mol_combo.clear()
        for mol in mols:
            name = (Path(mol.filepath).name
                    if mol.filepath else f"Molecule ({len(mol.elements)} atoms)")
            self.mol_combo.addItem(name)
        self.mol_combo.blockSignals(False)
        if 0 <= prev < self.mol_combo.count():
            self.mol_combo.setCurrentIndex(prev)
        if not mols:
            self.info_lbl.setText(
                "No molecules loaded. Load one first via Tools → Load molecule.")

    # ── Edit mode ─────────────────────────────────────────────────────────

    def _on_edit_toggled(self, active: bool):
        canvas = self._canvas()
        if canvas is None:
            self.edit_btn.setChecked(False)
            return
        mol = self._active_mol()
        if mol is None or len(mol.coordinates) == 0:
            self.edit_btn.setChecked(False)
            self.info_lbl.setText("No molecule selected or molecule has no atoms.")
            return

        self._edit_active = active
        if active:
            self.edit_btn.setText("Stop editing")
            self._press_cid = canvas.mpl_connect("button_press_event", self._on_press)
            self._motion_cid = canvas.mpl_connect("motion_notify_event", self._on_motion)
            self._release_cid = canvas.mpl_connect("button_release_event", self._on_release)
            self._draw_handles(mol, canvas)
            self.info_lbl.setText(
                "Click an atom to select it, then drag to move. Right-click = deselect.")
        else:
            self.edit_btn.setText("Start editing")
            self._disconnect_canvas(canvas)
            self._remove_handles(canvas)
            self._selected_idx = -1
            self.info_lbl.setText("Editing stopped.")

    def _disconnect_canvas(self, canvas=None):
        if canvas is None:
            canvas = self._canvas()
        if canvas is None:
            return
        for attr in ("_press_cid", "_motion_cid", "_release_cid"):
            cid = getattr(self, attr, None)
            if cid is not None:
                try:
                    canvas.mpl_disconnect(cid)
                except Exception:
                    pass
                setattr(self, attr, None)

    # ── Canvas event handlers ─────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes is None:
            return
        mol = self._active_mol()
        if mol is None:
            return

        if event.button == 3:
            self._selected_idx = -1
            self._press_canvas_xy = None
            self._draw_handles(mol, self._canvas())
            self.info_lbl.setText("Deselected.")
            return

        if event.button != 1:
            return

        tc = mol.get_transformed_coordinates()
        if len(tc) == 0:
            return

        canvas = self._canvas()
        ax = getattr(canvas, "main_ax", None)
        if ax is None:
            return

        # Compute nm-per-pixel for hit threshold (15 px)
        try:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            bbox = ax.get_window_extent()
            nm_per_px = max(
                abs(xlim[1] - xlim[0]) / max(bbox.width, 1),
                abs(ylim[1] - ylim[0]) / max(bbox.height, 1),
            )
        except Exception:
            nm_per_px = 0.05
        thresh = 15.0 * nm_per_px

        cx, cy = event.xdata, event.ydata
        best_idx, best_d2 = -1, thresh ** 2
        for i, (ax_nm, ay_nm, _) in enumerate(tc):
            d2 = (cx - ax_nm) ** 2 + (cy - ay_nm) ** 2
            if d2 < best_d2:
                best_d2, best_idx = d2, i

        if best_idx >= 0:
            self._selected_idx = best_idx
            self._press_canvas_xy = (cx, cy)
            self._press_atom_coords = mol.coordinates[best_idx].copy()
            elem = (mol.elements[best_idx]
                    if best_idx < len(mol.elements) else "?")
            lc = mol.coordinates[best_idx]
            self.info_lbl.setText(
                f"Selected: {elem} (atom {best_idx})  "
                f"({lc[0]:.2f}, {lc[1]:.2f}, {lc[2]:.2f}) Å  —  drag to move")
            self._draw_handles(mol, canvas)

    def _on_motion(self, event):
        if (event.inaxes is None or event.button != 1
                or self._selected_idx < 0
                or self._press_canvas_xy is None):
            return
        mol = self._active_mol()
        if mol is None:
            return

        dx_nm = event.xdata - self._press_canvas_xy[0]
        dy_nm = event.ydata - self._press_canvas_xy[1]

        M = _inv_transform_matrix(mol)
        d_local = M @ np.array([dx_nm, dy_nm, 0.0])

        mol.coordinates[self._selected_idx, 0] = (
            self._press_atom_coords[0] + d_local[0])
        mol.coordinates[self._selected_idx, 1] = (
            self._press_atom_coords[1] + d_local[1])

        canvas = self._canvas()
        self._draw_handles(mol, canvas)
        self._redraw(canvas)

    def _on_release(self, event):
        if event.button != 1:
            return
        mol = self._active_mol()
        if self._selected_idx >= 0 and mol is not None:
            elem = (mol.elements[self._selected_idx]
                    if self._selected_idx < len(mol.elements) else "?")
            lc = mol.coordinates[self._selected_idx]
            self.info_lbl.setText(
                f"Placed: {elem} (atom {self._selected_idx})  "
                f"({lc[0]:.2f}, {lc[1]:.2f}, {lc[2]:.2f}) Å")
        self._press_canvas_xy = None
        self._press_atom_coords = None

    # ── Handle drawing ────────────────────────────────────────────────────

    def _draw_handles(self, mol: Molecule, canvas):
        if canvas is None:
            return
        ax = getattr(canvas, "main_ax", None)
        if ax is None:
            return
        self._remove_handles(canvas, redraw=False)

        tc = mol.get_transformed_coordinates()
        for i, (ax_nm, ay_nm, _) in enumerate(tc):
            elem = mol.elements[i] if i < len(mol.elements) else "?"
            color = get_atom_color(elem, "cpk")
            selected = i == self._selected_idx
            artist, = ax.plot(
                [ax_nm], [ay_nm], "o",
                color=color,
                ms=14 if selected else 9,
                mec="#ffff00" if selected else "white",
                mew=2.5 if selected else 0.8,
                zorder=50,
                alpha=0.85,
            )
            self._handle_artists.append(artist)

        canvas.draw_idle()

    def _remove_handles(self, canvas=None, redraw=True):
        if canvas is None:
            canvas = self._canvas()
        for a in self._handle_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._handle_artists.clear()
        if redraw and canvas is not None:
            canvas.draw_idle()

    def _redraw(self, canvas):
        if canvas is None:
            return
        try:
            canvas._redraw()
        except Exception:
            canvas.draw_idle()

    # ── Actions ───────────────────────────────────────────────────────────

    def _relax_mmff(self):
        mol = self._active_mol()
        if mol is None or len(mol.coordinates) == 0:
            QtWidgets.QMessageBox.warning(self, "No molecule", "Select a molecule first.")
            return
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self, "RDKit missing",
                "RDKit is not installed.\nRun:  pip install rdkit")
            return
        self.info_lbl.setText("Running MMFF relaxation…")
        QtWidgets.QApplication.processEvents()
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mol", delete=False, mode="w")
            tmp.close()
            self._write_mol_v2000(mol, tmp.name)
            rdmol = Chem.MolFromMolFile(tmp.name, removeHs=False, sanitize=False)
            if rdmol is None:
                raise RuntimeError("RDKit could not parse the molecule.")
            try:
                Chem.SanitizeMol(rdmol)
            except Exception:
                pass
            result = AllChem.MMFFOptimizeMolecule(rdmol, maxIters=2000)
            if result == -1:
                raise RuntimeError("MMFF force field could not be set up.")
            conf = rdmol.GetConformer()
            new_coords = np.array([
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(rdmol.GetNumAtoms())
            ])
            center = np.mean(new_coords, axis=0)
            mol.coordinates = new_coords - center
            Path(tmp.name).unlink(missing_ok=True)

            canvas = self._canvas()
            if self._edit_active:
                self._draw_handles(mol, canvas)
            self._redraw(canvas)
            self.viewer.on_show_molecules_toggled(True)
            self.info_lbl.setText("MMFF relaxation done.")
        except Exception as e:
            self.info_lbl.setText(f"Relaxation failed: {e}")

    def _recalc_bonds(self):
        mol = self._active_mol()
        if mol is None:
            return
        mol.recalculate_bonds()
        canvas = self._canvas()
        if self._edit_active:
            self._draw_handles(mol, canvas)
        self._redraw(canvas)
        self.viewer.on_show_molecules_toggled(True)
        self.info_lbl.setText(f"Bonds recalculated: {len(mol.bonds)} bonds.")

    def _save_to_file(self):
        mol = self._active_mol()
        if mol is None or len(mol.coordinates) == 0:
            QtWidgets.QMessageBox.warning(self, "No molecule", "Select a molecule first.")
            return
        default = str(mol.filepath) if mol.filepath else "molecule.mol"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save molecule", default,
            "MOL files (*.mol);;All files (*)")
        if not path:
            return
        try:
            self._write_mol_v2000(mol, path)
            mol.filepath = path
            self._populate_molecules()
            self.info_lbl.setText(f"Saved to {Path(path).name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    @staticmethod
    def _write_mol_v2000(mol: Molecule, path: str):
        n_atoms = len(mol.coordinates)
        n_bonds = len(mol.bonds)
        lines = [
            Path(path).stem,
            "  SXMViewer",
            "",
            f"{n_atoms:3d}{n_bonds:3d}  0  0  0  0            999 V2000",
        ]
        for coord, elem in zip(mol.coordinates, mol.elements):
            lines.append(
                f"{coord[0]:10.4f}{coord[1]:10.4f}{coord[2]:10.4f}"
                f" {elem:<3s} 0  0  0  0  0  0  0  0  0  0  0  0")
        for i1, i2 in mol.bonds:
            lines.append(f"{i1+1:3d}{i2+1:3d}  1  0  0  0  0")
        lines.append("M  END")
        Path(path).write_text("\n".join(lines))

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        canvas = self._canvas()
        self._disconnect_canvas(canvas)
        self._remove_handles(canvas)
        super().closeEvent(event)
