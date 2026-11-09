"""Point defined lipid builder shared by Position monomers and Position coordinates.

The user adds a lipid row, switches Pick points on, clicks Start and End on the
image and then drags the square ends or the diamond handle to adjust the fit.
Geometry (head group and zig zag backbone) comes from MISO's ``lipid_building``;
the chain is forced through the bend point by ``_chain_through_mid``.

A host dialog mixes in ``LipidPickerMixin`` and provides:

    _lipid_canvas()           the matplotlib canvas (or None)
    _lipid_axes()             the image axes
    _lipid_redraw()           redraw the overlay of the host
    _lipid_artists()          list that receives the artists drawn here
    _lipid_release_other_pick()   switch the host's own pick modes off
    _ang_to_pixel(x, y)       Angstrom to image (col, row)
    _img, _px, _scan_dir      image array, pixel size in nm, "up" or "down"

and calls ``_build_lipid_group(layout)`` while building its UI.
"""
import sys
import time
from pathlib import Path

import numpy as np

from sxm_viewer._shared import QtCore, QtWidgets
from sxm_viewer.utils.miso_yaml import (
    LipidBond, MISOExportError, is_sugar, lipid_branch_parent, lipid_carbons)

# Colors used for built atoms (kept local so the module has no host dependency).
LIPID_ATOM_COLORS = {"C": "#9e9e9e", "O": "#e53935", "N": "#1e88e5", "H": "#eeeeee"}

DASH = "—"
ANGSTROM = "Å"


def import_lipid_engine():
    """Import MISO's lipid_building module (needs MISO/ on sys.path as a package)."""
    repo = Path(__file__).resolve().parents[3]
    miso = repo / "MISO"
    if not (miso / "src" / "rotation_optimization" / "structure" / "lipid_building.py").exists():
        raise ImportError(f"lipid_building.py not found under {miso}")
    if str(miso) not in sys.path:
        sys.path.insert(0, str(miso))
    from src.rotation_optimization.structure import lipid_building  # noqa: E402
    return lipid_building


# MISO build_linkage_headgroup linkage types (+ 'none' = plain hydrocarbon tail).
LINKAGES = ["none", "ester", "amide", "ether"]

# Ordered head group atoms per linkage: (main_chain_keys[], branch (key, main_index)).
# 'main' atoms bond sequentially and the last one bonds to the first backbone carbon;
# a branch atom bonds to the main atom at the given index. Element is read from the key.
_HEADGROUP_LAYOUT = {
    "ester":  (["o_ester_position", "c_carbonyl_position"], ("o_carbonyl_position", 1)),
    "amide":  (["n_link_position", "c_carbonyl_position"], ("o_carbonyl_position", 1)),
    "ether":  (["o_ether_position"], None),
    "none":   ([], None),
}


def _elem_from_key(key):
    return {"o": "O", "c": "C", "n": "N"}.get(key[0], "C")


class LipidPickerMixin:
    HANDLE_TOLERANCE_PX = 12.0
    C_C_BOND = 1.54

    # ------------------------------------------------------------------ state
    def _init_lipid_state(self):
        self._lipids = []         # parallel to lipid table rows
        self._lipid_engine = None
        self._lipid_cid = None
        self._lipid_stage = 0     # 0=click start, 1=click end, 2=adjust by dragging
        self._lipid_extra_cids = []   # motion/release connections while picking
        self._lipid_drag = None       # {"row", "key", "moved"} while a handle is dragged
        self._lipid_last_build = 0.0

    # ------------------------------------------------------------------ UI
    LIPID_HELP = (
        "How to draw a lipid. A lipid is a segment with two ends.\n"
        "Start: the FREE end of the chain (the tail tip).\n"
        "End: where the lipid is BONDED, either to a sugar or to another lipid chain. "
        "The connection happens at the End; MISO grows the chain from there toward the Start.\n"
        "1. Press Add lipid and switch Pick points on.\n"
        "2. Click the Start (free tip), then the End (bond point). One lipid at a time.\n"
        "3. Drag the square ends to adjust. Drag the diamond to bend the chain; Straighten "
        "removes the bend.\n"
        "4. Under MISO attachment choose a sugar and its carbon (put the End next to that "
        "sugar), or Branch off a lipid chain (put the End on that lipid's chain).\n"
        "5. Choose ester, amide or ether. Carbons 0 means automatic.\n"
        "A sugar carbon holds only one thing: a sugar bond, a peptide bond or a lipid.\n"
        "Branches: MISO only builds a branch with the amide linkage (ester and ether "
        "branches get no chain), so branches are set to amide. A branch needs at least one "
        "lipid attached to a sugar, all on one sugar, and all branches on the same lipid. "
        "MISO bonds a branch to carbon 2 of that lipid, so put the branch End point near "
        "the beginning of that lipid's chain, next to its sugar.")

    def _build_lipid_group(self, parent_layout):
        self.lipid_group = QtWidgets.QGroupBox("Lipids")
        self.lipid_group.setVisible(False)
        lg = QtWidgets.QVBoxLayout(self.lipid_group)
        self.lipid_help = QtWidgets.QLabel(self.LIPID_HELP)
        self.lipid_help.setWordWrap(True)
        self.lipid_help.setStyleSheet("color: #455a64;")
        lg.addWidget(self.lipid_help)
        self.lipid_table = QtWidgets.QTableWidget(0, 6)
        self.lipid_table.setHorizontalHeaderLabels(
            [f"Start, free tip ({ANGSTROM})", f"End, bond point ({ANGSTROM})",
             f"Mid ({ANGSTROM})",
             "Carbons (0=auto)", "Linkage", "Attached to"])
        lhdr = self.lipid_table.horizontalHeader()
        for c in range(3):
            lhdr.setSectionResizeMode(c, QtWidgets.QHeaderView.Stretch)
        for c in (3, 4, 5):
            lhdr.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        self.lipid_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.lipid_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.lipid_table.setFixedHeight(130)
        self.lipid_table.currentCellChanged.connect(
            lambda cr, cc, pr, pc: self._on_lipid_selected(cr))
        lg.addWidget(self.lipid_table)

        lbtn = QtWidgets.QHBoxLayout()
        add_lip = QtWidgets.QPushButton("Add lipid")
        add_lip.clicked.connect(self._add_lipid_row)
        rm_lip = QtWidgets.QPushButton("Remove lipid")
        rm_lip.clicked.connect(self._remove_lipid_row)
        self.lipid_pick_btn = QtWidgets.QPushButton("Pick points: OFF")
        self.lipid_pick_btn.setCheckable(True)
        self.lipid_pick_btn.toggled.connect(self._on_lipid_pick_toggled)
        clr_lip = QtWidgets.QPushButton("Clear points")
        clr_lip.clicked.connect(self._clear_lipid_points)
        straight_lip = QtWidgets.QPushButton("Straighten")
        straight_lip.setToolTip("Remove the bend (the Mid point) of the selected lipid.")
        straight_lip.clicked.connect(self._straighten_lipid)
        lbtn.addWidget(add_lip)
        lbtn.addWidget(rm_lip)
        lbtn.addWidget(self.lipid_pick_btn)
        lbtn.addWidget(clr_lip)
        lbtn.addWidget(straight_lip)
        lbtn.addStretch()
        lg.addLayout(lbtn)
        self.lipid_status = QtWidgets.QLabel(
            "Add a lipid, switch Pick points on, click Start (free tip) then End "
            "(bond point), then drag the handles to adjust.")
        self.lipid_status.setWordWrap(True)
        lg.addWidget(self.lipid_status)

        # How the selected lipid joins the assembly in the MISO YAML.
        attach = QtWidgets.QHBoxLayout()
        attach.addWidget(QtWidgets.QLabel("MISO attachment:"))
        self.lipid_anchor_combo = QtWidgets.QComboBox()
        self.lipid_anchor_combo.addItem("Sugar carbon", "sugar")
        self.lipid_anchor_combo.addItem("Branch off a lipid chain", "lipid")
        self.lipid_anchor_combo.setToolTip(
            "A lipid starts at a sugar carbon, or branches from the chain of another "
            "lipid at its Start point (MISO connection_anchor).")
        self.lipid_attach_sugar = QtWidgets.QComboBox()
        self.lipid_attach_carbon = QtWidgets.QComboBox()
        for widget in (self.lipid_anchor_combo, self.lipid_attach_sugar,
                       self.lipid_attach_carbon):
            attach.addWidget(widget)
        attach.addStretch()
        lg.addLayout(attach)
        self.lipid_anchor_combo.currentIndexChanged.connect(self._on_lipid_attach_changed)
        self.lipid_attach_sugar.currentIndexChanged.connect(self._on_lipid_attach_sugar_changed)
        self.lipid_attach_carbon.currentIndexChanged.connect(self._on_lipid_attach_changed)
        self._refresh_lipid_attach_ui()
        parent_layout.addWidget(self.lipid_group)

    # ------------------------------------------------------------------ rows
    def _on_include_lipids_toggled(self, checked):
        self.lipid_group.setVisible(bool(checked))
        if not checked and self.lipid_pick_btn.isChecked():
            self.lipid_pick_btn.setChecked(False)

    def _add_lipid_row(self):
        r = self.lipid_table.rowCount()
        self.lipid_table.insertRow(r)
        for c in range(3):
            it = QtWidgets.QTableWidgetItem(DASH)
            it.setTextAlignment(QtCore.Qt.AlignCenter)
            self.lipid_table.setItem(r, c, it)
        carbons = QtWidgets.QSpinBox()
        carbons.setRange(0, 999)      # 0 = auto (estimate from distance)
        carbons.setValue(0)
        carbons.valueChanged.connect(self._on_lipid_param_changed)
        self.lipid_table.setCellWidget(r, 3, carbons)
        linkage = QtWidgets.QComboBox()
        linkage.addItems(LINKAGES)
        linkage.currentTextChanged.connect(self._on_lipid_param_changed)
        self.lipid_table.setCellWidget(r, 4, linkage)
        self._lipids.append({
            "label": f"L{r+1}", "start": None, "end": None, "mid": None,
            "carbons": None, "linkage": "none",
            "anchor": "sugar", "sugar": None, "carbon": None,
            "atoms": None, "atom_types": None, "bonds": None,
        })
        it = QtWidgets.QTableWidgetItem(DASH)
        it.setTextAlignment(QtCore.Qt.AlignCenter)
        self.lipid_table.setItem(r, 5, it)
        self.lipid_table.setCurrentCell(r, 0)

    def _remove_lipid_row(self):
        r = self.lipid_table.currentRow()
        if r < 0:
            return
        self.lipid_table.removeRow(r)
        del self._lipids[r]
        # Re-label remaining lipids to stay in sync with row order.
        for i, lip in enumerate(self._lipids):
            lip["label"] = f"L{i+1}"
        self._lipid_redraw()

    def _on_lipid_selected(self, row):
        self._lipid_drag = None
        self._sync_lipid_attach_combos()
        if self.lipid_pick_btn.isChecked():
            self._update_lipid_status()
        self._lipid_redraw()

    def _lipid_row_of_widget(self, widget):
        for r in range(self.lipid_table.rowCount()):
            if (self.lipid_table.cellWidget(r, 3) is widget
                    or self.lipid_table.cellWidget(r, 4) is widget):
                return r
        return None

    def _on_lipid_param_changed(self, *_args):
        # Resolve the row from the sender so removals can't stale the index.
        row = self._lipid_row_of_widget(self.sender())
        if row is None or not (0 <= row < len(self._lipids)):
            return
        cw = self.lipid_table.cellWidget(row, 3)
        val = cw.value() if cw else 0
        self._lipids[row]["carbons"] = val if val > 0 else None
        lw = self.lipid_table.cellWidget(row, 4)
        self._lipids[row]["linkage"] = lw.currentText() if lw else "none"
        self._rebuild_lipid_geometry(self._lipids[row])
        self._lipid_redraw()

    @staticmethod
    def _lipid_stage_of(lip):
        """0: click Start, 1: click End, 2: both set, adjust by dragging handles."""
        if lip is None or lip.get("start") is None:
            return 0
        return 1 if lip.get("end") is None else 2

    def _selected_lipid(self):
        r = self.lipid_table.currentRow()
        return self._lipids[r] if 0 <= r < len(self._lipids) else None

    # ------------------------------------------------------------------ MISO attachment
    def _lipid_sugar_units(self):
        return [(i, inst) for i, inst in enumerate(getattr(self, "_instances", []))
                if is_sugar(inst)]

    def _lipid_attach_text(self, lip):
        if lip.get("anchor") == "lipid":
            return "branch (at End point)"
        index, carbon = lip.get("sugar"), lip.get("carbon")
        instances = getattr(self, "_instances", [])
        if index is None or carbon is None or not 0 <= index < len(instances):
            return DASH
        return f"{instances[index]['label']} {carbon}"

    def _update_lipid_attach_cell(self, row):
        item = self.lipid_table.item(row, 5)
        if item is not None and 0 <= row < len(self._lipids):
            item.setText(self._lipid_attach_text(self._lipids[row]))

    def _refresh_lipid_attach_ui(self):
        """Refill the sugar list after units change and drop attachments that no longer fit."""
        instances = getattr(self, "_instances", [])
        for row, lip in enumerate(self._lipids):
            index = lip.get("sugar")
            if lip.get("anchor", "sugar") == "sugar" and index is not None:
                ok = (0 <= index < len(instances) and is_sugar(instances[index])
                      and lip.get("carbon") in lipid_carbons(instances[index]))
                if not ok:
                    lip["sugar"] = lip["carbon"] = None
            self._update_lipid_attach_cell(row)
        blockers = [QtCore.QSignalBlocker(self.lipid_attach_sugar)]
        self.lipid_attach_sugar.clear()
        self.lipid_attach_sugar.addItem("Choose sugar...", None)
        for index, inst in self._lipid_sugar_units():
            self.lipid_attach_sugar.addItem(inst["label"], index)
        del blockers
        self._sync_lipid_attach_combos()

    def _fill_lipid_attach_carbons(self):
        index = self.lipid_attach_sugar.currentData()
        blocker = QtCore.QSignalBlocker(self.lipid_attach_carbon)
        self.lipid_attach_carbon.clear()
        if index is not None and 0 <= index < len(self._instances):
            self.lipid_attach_carbon.addItems(lipid_carbons(self._instances[index]))
        del blocker

    def _sync_lipid_attach_combos(self):
        """Show the selected lipid's attachment in the three controls."""
        lip = self._selected_lipid()
        blockers = [QtCore.QSignalBlocker(w) for w in
                    (self.lipid_anchor_combo, self.lipid_attach_sugar, self.lipid_attach_carbon)]
        enabled = lip is not None
        self.lipid_anchor_combo.setEnabled(enabled)
        if lip is None:
            self.lipid_attach_sugar.setCurrentIndex(0)
            self._fill_lipid_attach_carbons()
        else:
            self.lipid_anchor_combo.setCurrentIndex(
                max(0, self.lipid_anchor_combo.findData(lip.get("anchor", "sugar"))))
            self.lipid_attach_sugar.setCurrentIndex(
                max(0, self.lipid_attach_sugar.findData(lip.get("sugar"))))
            self._fill_lipid_attach_carbons()
            if lip.get("carbon"):
                self.lipid_attach_carbon.setCurrentIndex(
                    max(0, self.lipid_attach_carbon.findText(lip["carbon"])))
        del blockers
        on_sugar = enabled and self.lipid_anchor_combo.currentData() == "sugar"
        self.lipid_attach_sugar.setEnabled(on_sugar)
        self.lipid_attach_carbon.setEnabled(on_sugar)

    def _on_lipid_attach_sugar_changed(self, _index=None):
        self._fill_lipid_attach_carbons()
        self._on_lipid_attach_changed()

    def _on_lipid_attach_changed(self, _index=None):
        row = self.lipid_table.currentRow()
        if not (0 <= row < len(self._lipids)):
            return
        lip = self._lipids[row]
        lip["anchor"] = self.lipid_anchor_combo.currentData() or "sugar"
        if lip["anchor"] == "sugar":
            lip["sugar"] = self.lipid_attach_sugar.currentData()
            lip["carbon"] = (self.lipid_attach_carbon.currentText() or None
                             if lip["sugar"] is not None else None)
        else:
            lip["sugar"] = lip["carbon"] = None
            lip["linkage"] = "amide"             # MISO only builds amide branches
            combo = self.lipid_table.cellWidget(row, 4)
            if combo is not None:
                blocker = QtCore.QSignalBlocker(combo)
                combo.setCurrentText("amide")
                del blocker
        on_sugar = lip["anchor"] == "sugar"
        self.lipid_attach_sugar.setEnabled(on_sugar)
        self.lipid_attach_carbon.setEnabled(on_sugar)
        self._update_lipid_attach_cell(row)

    def _lipid_sites(self):
        """(sugar index, carbon) pairs already used by lipids, so bonds avoid them."""
        return {(lip["sugar"], lip["carbon"]) for lip in self._lipids
                if lip.get("anchor") == "sugar" and lip.get("sugar") is not None
                and lip.get("carbon")}

    @staticmethod
    def _point_tuple(point):
        return None if point is None else (float(point[0]), float(point[1]))

    def _lipid_bonds(self):
        """The lipid rows as LipidBond objects for the YAML builder."""
        return [LipidBond(
            label=lip["label"], anchor=lip.get("anchor", "sugar"),
            sugar=lip.get("sugar"), sugar_carbon=lip.get("carbon"),
            start=self._point_tuple(lip.get("start")), end=self._point_tuple(lip.get("end")),
            mid=self._point_tuple(lip.get("mid")), carbons=lip.get("carbons"),
            linkage=lip.get("linkage", "none")) for lip in self._lipids]

    def _lipid_context(self):
        """Allowlisted description of the lipid rows for the assistant."""
        rows = []
        for number, lip in enumerate(self._lipids):
            rows.append({
                "lipid": number, "label": lip["label"],
                "start_xy_angstrom": self._point_tuple(lip.get("start")),
                "end_xy_angstrom": self._point_tuple(lip.get("end")),
                "mid_xy_angstrom": self._point_tuple(lip.get("mid")),
                "has_bend": lip.get("mid") is not None,
                "carbons": lip.get("carbons"), "linkage": lip.get("linkage", "none"),
                "anchor": lip.get("anchor", "sugar"),
                "sugar": lip.get("sugar"), "sugar_carbon": lip.get("carbon")})
        return rows

    def _lipid_yaml_note(self):
        """Text for the export dialog when lipids are part of the YAML."""
        if not self._lipids:
            return ""
        note = f"\n\nLipids written to the YAML: {len(self._lipids)}."
        try:
            parent = lipid_branch_parent(self._lipid_bonds())
        except MISOExportError:
            parent = None
        if parent is not None:
            note += (f"\nBranches bond to carbon 2 of {self._lipids[parent]['label']} "
                     "(MISO uses carbon 2 of the first lipid it builds).")
        if any(lip.get("mid") is not None for lip in self._lipids):
            note += ("\nA bend (midpoint) is written, but MISO's chain builder may pull the "
                     "chain off the bend; check the result.")
        return note

    def _lipid_assistant_blocker(self):
        """Why the assistant cannot take the lipid rows yet, or None."""
        for lip in self._lipids:
            if lip.get("start") is None or lip.get("end") is None:
                return (f"Lipid {lip['label']} needs a Start and an End point. Pick them "
                        "(or remove the row) before using the assistant.")
        return None

    def _apply_lipid_attachments(self, attachments):
        """Write assistant (or wizard) attachments into the lipid rows."""
        for att in attachments:
            if not 0 <= att.lipid < len(self._lipids):
                continue
            lip = self._lipids[att.lipid]
            lip["anchor"] = att.anchor
            lip["sugar"] = att.sugar if att.anchor == "sugar" else None
            lip["carbon"] = att.sugar_carbon if att.anchor == "sugar" else None
            if att.linkage:
                lip["linkage"] = att.linkage
                combo = self.lipid_table.cellWidget(att.lipid, 4)
                if combo is not None:
                    blocker = QtCore.QSignalBlocker(combo)
                    combo.setCurrentText(att.linkage)
                    del blocker
            self._update_lipid_attach_cell(att.lipid)
        self._sync_lipid_attach_combos()

    # ------------------------------------------------------------------ picking
    def _on_lipid_pick_toggled(self, active):
        canvas = self._lipid_canvas()
        if canvas is None:
            self.lipid_pick_btn.setChecked(False)
            return
        if active:
            # Only one canvas pick mode at a time.
            self._lipid_release_other_pick()
            if self.lipid_table.currentRow() < 0 and self._lipids:
                self.lipid_table.setCurrentCell(0, 0)
            self._lipid_stage = self._lipid_stage_of(self._selected_lipid())
            self._lipid_cid = canvas.mpl_connect("button_press_event", self._on_lipid_canvas_click)
            self._lipid_extra_cids = [
                canvas.mpl_connect("motion_notify_event", self._on_lipid_canvas_motion),
                canvas.mpl_connect("button_release_event", self._on_lipid_canvas_release)]
            self.lipid_pick_btn.setText("Pick points: ON")
            self._update_lipid_status()
        else:
            self._disconnect_lipid_events(canvas)
            self.lipid_pick_btn.setText("Pick points: OFF")
        self._lipid_redraw()

    def _disconnect_lipid_events(self, canvas=None):
        canvas = canvas or self._lipid_canvas()
        for cid in [self._lipid_cid, *self._lipid_extra_cids]:
            if cid is not None and canvas is not None:
                canvas.mpl_disconnect(cid)
        self._lipid_cid = None
        self._lipid_extra_cids = []
        self._lipid_drag = None

    def _update_lipid_status(self):
        r = self.lipid_table.currentRow()
        lip = self._lipids[r] if 0 <= r < len(self._lipids) else None
        who = lip["label"] if lip is not None else "?"
        self._lipid_stage = self._lipid_stage_of(lip)
        text = [f"{who}: click the Start point, the FREE tip of the chain.",
                f"{who}: click the End point, where the lipid is bonded to a sugar or a lipid.",
                f"{who}: drag the square ends or the diamond handle to adjust the fit. "
                "Dragging the diamond bends the chain; 'Straighten' removes the bend."][
                    self._lipid_stage]
        self.lipid_status.setText(text)

    def _clear_lipid_points(self):
        r = self.lipid_table.currentRow()
        if not (0 <= r < len(self._lipids)):
            return
        lip = self._lipids[r]
        lip["start"] = lip["end"] = lip["mid"] = None
        lip["atoms"] = lip["atom_types"] = lip["bonds"] = None
        for c in range(3):
            self.lipid_table.item(r, c).setText(DASH)
        self._update_lipid_status()
        self._lipid_redraw()

    def _straighten_lipid(self):
        lip = self._selected_lipid()
        if lip is None or lip["mid"] is None:
            return
        lip["mid"] = None
        self.lipid_table.item(self.lipid_table.currentRow(), 2).setText(DASH)
        self._rebuild_lipid_geometry(lip)
        self._update_lipid_status()
        self._lipid_redraw()

    def _reset_lipids(self):
        """Drop every lipid row (used when the host reloads its image)."""
        if self.lipid_pick_btn.isChecked():
            self.lipid_pick_btn.setChecked(False)
        self.lipid_table.setRowCount(0)
        self._lipids = []
        self._lipid_drag = None

    # ------------------------------------------------------------------ geometry of the picks
    def _event_to_ang(self, event):
        """Event position in Angstrom, not snapped to the pixel grid."""
        H, W = self._img.shape
        col = min(max(float(event.xdata), 0.0), W - 1.0)
        row = min(max(float(event.ydata), 0.0), H - 1.0)
        orig_y = (row + 1.0) if self._scan_dir == "down" else (H - 1.0 - row)
        return [col * self._px[0] * 10.0, orig_y * self._px[1] * 10.0]

    def _lipid_handles(self, lip):
        """{key: (x_ang, y_ang)} of the draggable points; the bend handle defaults
        to the middle of Start and End until the user drags it."""
        handles = {}
        if lip.get("start") is not None:
            handles["start"] = lip["start"]
        if lip.get("end") is not None:
            handles["end"] = lip["end"]
        if "start" in handles and "end" in handles:
            handles["mid"] = lip["mid"] if lip.get("mid") is not None else [
                (lip["start"][0] + lip["end"][0]) / 2.0, (lip["start"][1] + lip["end"][1]) / 2.0]
        return handles

    def _lipid_handle_at(self, lip, event):
        """Key of the handle under the cursor (screen distance), or None."""
        ax = self._lipid_axes()
        best, best_distance = None, self.HANDLE_TOLERANCE_PX
        for key, (x_ang, y_ang) in self._lipid_handles(lip).items():
            col, row = self._ang_to_pixel(x_ang, y_ang)
            sx, sy = ax.transData.transform((col, row))
            distance = float(np.hypot(sx - event.x, sy - event.y))
            if distance <= best_distance:
                best, best_distance = key, distance
        return best

    def _set_lipid_point(self, row, key, point):
        lip = self._lipids[row]
        lip[key] = point
        column = ("start", "end", "mid").index(key)
        self.lipid_table.item(row, column).setText(f"{point[0]:.2f}, {point[1]:.2f}")

    def _on_lipid_canvas_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        r = self.lipid_table.currentRow()
        if not (0 <= r < len(self._lipids)):
            QtWidgets.QMessageBox.information(self, "No lipid", "Select a lipid row first.")
            return
        lip = self._lipids[r]
        key = self._lipid_handle_at(lip, event)
        if key is not None:                      # grab a handle to adjust the fit
            self._lipid_drag = {"row": r, "key": key, "moved": False}
            return
        stage = self._lipid_stage_of(lip)
        if stage >= 2:
            self._update_lipid_status()
            return
        self._set_lipid_point(r, ("start", "end")[stage], self._event_to_ang(event))
        self._rebuild_lipid_geometry(lip)
        self._update_lipid_status()
        self._lipid_redraw()

    def _on_lipid_canvas_motion(self, event):
        drag = self._lipid_drag
        if drag is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        drag["moved"] = True
        self._set_lipid_point(drag["row"], drag["key"], self._event_to_ang(event))
        lip = self._lipids[drag["row"]]
        now = time.monotonic()
        if now - self._lipid_last_build >= 0.06:      # live fit, throttled
            self._lipid_last_build = now
            self._rebuild_lipid_geometry(lip)
        self._lipid_redraw()

    def _on_lipid_canvas_release(self, event):
        drag, self._lipid_drag = self._lipid_drag, None
        if drag is None:
            return
        if drag["moved"]:
            self._rebuild_lipid_geometry(self._lipids[drag["row"]])
            self._update_lipid_status()
            self._lipid_redraw()

    # ------------------------------------------------------------------ building
    def _chain_through_mid(self, L, start, mid, end, n_carbons=None):
        """Backbone that really passes through ``mid``.

        MISO's two segment builder places the chain through the waypoint and then
        runs one FABRIK pass towards ``end`` only, which drags the chain off the
        waypoint. Here each half is fitted on its own: Start to Mid, then from where
        the first half ended to End. Long enough halves reach their target exactly.
        """
        start, mid, end = (np.asarray(p, dtype=float) for p in (start, mid, end))
        d1 = float(np.linalg.norm(mid - start))
        d2 = float(np.linalg.norm(end - mid))
        if n_carbons:
            total = max(4, int(n_carbons))
            n1 = max(2, int(round(total * d1 / max(d1 + d2, 1e-9))))
            n2 = max(2, total - n1)
        else:
            n1 = max(2, int(np.ceil(d1 / self.C_C_BOND)))
            n2 = max(2, int(np.ceil(d2 / self.C_C_BOND)))
        first = [np.asarray(c, dtype=float) for c in
                 L.build_lipid_backbone(start, mid, None, use_fabrik=True,
                                        n_carbons=n1)["chain_carbons"]]
        anchor = first[-1]
        second = [np.asarray(c, dtype=float) for c in
                  L.build_lipid_backbone(anchor, end, None, use_fabrik=True,
                                         n_carbons=n2)["chain_carbons"]]
        return first + second

    def _rebuild_lipid_geometry(self, lip):
        """Build head group + zig zag backbone via MISO lipid_building."""
        lip["atoms"] = lip["atom_types"] = lip["bonds"] = None
        if lip["start"] is None or lip["end"] is None:
            return
        if self._lipid_engine is None:
            try:
                self._lipid_engine = self._import_lipid_engine()
            except Exception as exc:
                self.lipid_status.setText(f"Lipid engine import failed: {exc}")
                return
        L = self._lipid_engine
        # The lipid is bonded at its End and grows toward its free Start, as in MISO.
        start = np.array([lip["end"][0], lip["end"][1], 0.0])
        end = np.array([lip["start"][0], lip["start"][1], 0.0])
        mid = np.array([lip["mid"][0], lip["mid"][1], 0.0]) if lip["mid"] else None
        linkage = lip.get("linkage", "none")
        n_carbons = lip.get("carbons")

        atoms, atom_types, bonds = [], [], []
        aim = mid if mid is not None else end
        direction = aim - start
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return
        direction = direction / norm

        cur = start
        main_last_idx = None
        try:
            if linkage != "none":
                hg, cur = L.build_linkage_headgroup(start, direction, linkage)
                main_keys, branch = _HEADGROUP_LAYOUT[linkage]
                for key in main_keys:
                    atoms.append(np.asarray(hg[key], dtype=float))
                    atom_types.append(_elem_from_key(key))
                    if main_last_idx is not None:
                        bonds.append((main_last_idx, len(atoms) - 1))
                    main_last_idx = len(atoms) - 1
                if branch is not None:
                    bkey, bmain = branch
                    atoms.append(np.asarray(hg[bkey], dtype=float))
                    atom_types.append(_elem_from_key(bkey))
                    bonds.append((bmain, len(atoms) - 1))
            if mid is None:
                if not n_carbons:
                    # MISO's own estimate undershoots on a straight pick, so the chain
                    # stops short of End; size it from the real remaining distance.
                    n_carbons = max(2, int(np.ceil(
                        float(np.linalg.norm(end - np.asarray(cur, dtype=float))) / self.C_C_BOND)))
                bb = L.build_lipid_backbone(cur, end, None, use_fabrik=True, n_carbons=n_carbons)
                chain = [np.asarray(c, dtype=float) for c in bb["chain_carbons"]]
            else:
                chain = self._chain_through_mid(L, cur, mid, end, n_carbons)
        except Exception as exc:
            self.lipid_status.setText(f"Lipid build failed: {exc}")
            return

        first_c_idx = len(atoms)
        for i, c in enumerate(chain):
            atoms.append(c)
            atom_types.append("C")
            if i > 0:
                bonds.append((first_c_idx + i - 1, first_c_idx + i))
        if main_last_idx is not None and chain:
            bonds.append((main_last_idx, first_c_idx))   # head group to first carbon

        lip["atoms"] = np.array(atoms)
        lip["atom_types"] = atom_types
        lip["bonds"] = bonds

    @staticmethod
    def _import_lipid_engine():
        return import_lipid_engine()

    # ------------------------------------------------------------------ drawing
    def _draw_lipids(self, ax, artists=None, atom_colors=None):
        """Draw handles, picked points and built chains. Artists are appended to
        ``artists`` (default: the host's list) so the host can clear them later."""
        artists = self._lipid_artists() if artists is None else artists
        atom_colors = atom_colors or LIPID_ATOM_COLORS
        sel = self.lipid_table.currentRow() if hasattr(self, "lipid_table") else -1
        for idx, lip in enumerate(self._lipids):
            is_sel = idx == sel
            if is_sel and self.lipid_pick_btn.isChecked():
                handles = self._lipid_handles(lip)
                if "mid" in handles:
                    guide = [self._ang_to_pixel(*handles[k]) for k in ("start", "mid", "end")]
                    g, = ax.plot([p[0] for p in guide], [p[1] for p in guide], "--",
                                 color="#ffe082", lw=1.0, zorder=23, alpha=0.9)
                    artists.append(g)
                    if lip.get("mid") is None:      # draggable bend handle, not yet used
                        c, rw = guide[1]
                        h, = ax.plot([c], [rw], marker="D", mfc="white", mec="#fb8c00",
                                     mew=1.6, ms=11, zorder=26)
                        artists.append(h)
            # start / end / mid markers
            for key, color in (("start", "#43a047"), ("end", "#e53935"), ("mid", "#fb8c00")):
                p = lip.get(key)
                if p is None:
                    continue
                c, rw = self._ang_to_pixel(p[0], p[1])
                m, = ax.plot([c], [rw], marker="s" if key != "mid" else "D",
                             color=color, ms=8 if is_sel else 6,
                             mec="white", mew=0.7, zorder=24)
                artists.append(m)
                if key in ("start", "end"):
                    caption = lip["label"] if key == "start" else ""
                    if is_sel:
                        caption = f"{lip['label']} " + ("tip" if key == "start" else "bond")
                    if caption:
                        t = ax.annotate(caption, xy=(c, rw), xytext=(5, 5),
                                        textcoords="offset points", color="#c8e6c9",
                                        fontsize=8, zorder=25)
                        artists.append(t)
            # built chain
            atoms = lip.get("atoms")
            if atoms is None or len(atoms) == 0:
                continue
            px = {i: self._ang_to_pixel(a[0], a[1]) for i, a in enumerate(atoms)}
            for (a, b) in lip.get("bonds", []):
                (ca, ra), (cb, rb) = px[a], px[b]
                ln, = ax.plot([ca, cb], [ra, rb], "-",
                              color="#26c6da" if is_sel else "#78909c",
                              lw=1.6 if is_sel else 1.0, zorder=17, alpha=0.9)
                artists.append(ln)
            cols = [px[i][0] for i in range(len(atoms))]
            rows = [px[i][1] for i in range(len(atoms))]
            colors = [atom_colors.get(s, "#8e24aa") for s in lip["atom_types"]]
            sc = ax.scatter(cols, rows, c=colors, s=18,
                            edgecolors="white", linewidths=0.3, zorder=19, alpha=0.95)
            artists.append(sc)

    def _draw_lipids_export(self, ax, atom_colors=None):
        """Static drawing of the built lipids on a fresh export figure."""
        atom_colors = atom_colors or LIPID_ATOM_COLORS
        for lip in self._lipids:
            atoms = lip.get("atoms")
            if atoms is None or len(atoms) == 0:
                continue
            px = {i: self._ang_to_pixel(a[0], a[1]) for i, a in enumerate(atoms)}
            for (a, b) in lip.get("bonds", []):
                (ca, ra), (cb, rb) = px[a], px[b]
                ax.plot([ca, cb], [ra, rb], "-", color="#78909c", lw=0.9, zorder=17)
            for i, sym in enumerate(lip["atom_types"]):
                c, rw = px[i]
                ax.plot(c, rw, marker="o", ms=3.0,
                        color=atom_colors.get(sym, "#8e24aa"), mec="white", mew=0.3, zorder=19)
            for key, color in (("start", "#43a047"), ("end", "#e53935"), ("mid", "#fb8c00")):
                p = lip.get(key)
                if p is None:
                    continue
                c, rw = self._ang_to_pixel(p[0], p[1])
                ax.plot(c, rw, marker="s", color=color, ms=6, mec="white", mew=0.5, zorder=24)
            if lip.get("start"):
                sc, sr = self._ang_to_pixel(lip["start"][0], lip["start"][1])
                ax.annotate(lip["label"], xy=(sc, sr), xytext=(4, 4),
                            textcoords="offset points", color="#c8e6c9", fontsize=7, zorder=25)

    # ------------------------------------------------------------------ export
    def _placed_lipids(self):
        return [lip for lip in self._lipids if lip.get("atoms") is not None]

    def _export_lipids_csv(self, out_path, placed_lipids):
        """Write MISO shaped lipid params + a companion per atom coordinate file."""
        import csv
        stem = Path(out_path).with_suffix("")
        params_path = f"{stem}_lipids.csv"
        atoms_path = f"{stem}_lipids_atoms.csv"

        def fmt(p):
            return (f"{p[0]:.6f}", f"{p[1]:.6f}") if p else ("", "")

        with open(params_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Lipid", "Start_X", "Start_Y", "End_X", "End_Y",
                        "Mid_X", "Mid_Y", "Carbons", "Linkage", "N_built_atoms",
                        "Anchor", "Attached_to"])
            for lip in placed_lipids:
                sx, sy = fmt(lip["start"])
                ex, ey = fmt(lip["end"])
                mx, my = fmt(lip["mid"])
                carbons = lip["carbons"] if lip["carbons"] else "auto"
                w.writerow([lip["label"], sx, sy, ex, ey, mx, my,
                            carbons, lip["linkage"], len(lip["atoms"]),
                            lip.get("anchor", "sugar"),
                            self._lipid_attach_text(lip).replace(DASH, "")])

        with open(atoms_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Lipid", "Atom_Index", "Element",
                        "X (Angstrom)", "Y (Angstrom)", "Z (Angstrom)"])
            for lip in placed_lipids:
                for i, (xyz, sym) in enumerate(zip(lip["atoms"], lip["atom_types"])):
                    w.writerow([lip["label"], i, sym,
                                f"{xyz[0]:.6f}", f"{xyz[1]:.6f}", f"{xyz[2]:.6f}"])
        return [params_path, atoms_path]
