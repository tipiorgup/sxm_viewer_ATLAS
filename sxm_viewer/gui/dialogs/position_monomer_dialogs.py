from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

from ..._shared import QtCore, QtWidgets
from ._sxm_image_loader import load_demo_image


# Ring-type options wired to monomer_building.classify_puckering base labels.
# The engine filters on descriptor.split('_')[0], so these bases match
# chair_4C1/chair_1C4, boat/boat_intermediate, twist_intermediate,
# envelope_N, half_chair.
RING_TYPES = ["Any", "chair", "boat", "twist", "envelope", "half"]
ANOMERS = ["Any", "alpha", "beta"]

# Amino-acid tables mirrored from MISO peptide_building.build_peptide_with_rdkit_ca
# (those dicts are function-local there, so we replicate the standard 20 here).
AA_MAP = {'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
          'E': 'Glu', 'Q': 'Gln', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
          'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
          'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'}
AA_SMILES = {
    'Ala': 'N[C@@H](C)C(=O)O',
    'Val': 'N[C@@H](C(C)C)C(=O)O',
    'Leu': 'N[C@@H](CC(C)C)C(=O)O',
    'Ile': 'N[C@@H]([C@H](CC)C)C(=O)O',
    'Met': 'N[C@@H](CCSC)C(=O)O',
    'Phe': 'N[C@@H](Cc1ccccc1)C(=O)O',
    'Trp': 'N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O',
    'Pro': 'N1[C@@H](CCC1)C(=O)O',
    'Ser': 'N[C@@H](CO)C(=O)O',
    'Thr': 'N[C@@H]([C@H](O)C)C(=O)O',
    'Tyr': 'N[C@@H](Cc1ccc(O)cc1)C(=O)O',
    'Cys': 'N[C@@H](CS)C(=O)O',
    'Asn': 'N[C@@H](CC(=O)N)C(=O)O',
    'Gln': 'N[C@@H](CCC(=O)N)C(=O)O',
    'Asp': 'N[C@@H](CC(=O)O)C(=O)O',
    'Glu': 'N[C@@H](CCC(=O)O)C(=O)O',
    'Lys': 'N[C@@H](CCCCN)C(=O)O',
    'Arg': 'N[C@@H](CCCNC(=N)N)C(=O)O',
    'His': 'N[C@@H](Cc1c[nH]cn1)C(=O)O',
    'Gly': 'NCC(=O)O',
}


def resolve_aa(code):
    """Map a 1- or 3-letter residue code to (canonical_name, smiles).

    Accepts 'N', 'Asn', 'ASN' (case-insensitive on 3-letter). Returns
    (None, None) when the code is not one of the standard 20.
    """
    code = (code or "").strip()
    if not code:
        return None, None
    if len(code) == 1:
        name = AA_MAP.get(code.upper())
    else:
        name = code.title()
        if name not in AA_SMILES:
            name = None
    if name is None:
        return None, None
    return name, AA_SMILES[name]

# Element colors for the 2D overlay (top-down projection onto the STM image).
ATOM_COLORS = {"C": "#222222", "O": "#e53935", "N": "#1e88e5",
               "H": "#bbbbbb", "S": "#fdd835", "P": "#fb8c00"}
ATOM_SIZE = {"C": 26, "O": 24, "N": 24, "H": 8, "S": 30, "P": 30}


# --------------------------------------------------------------------------- math
def euler_to_matrix(rx, ry, rz):
    """Rotation matrix for extrinsic X→Y→Z rotations (degrees). R = Rz@Ry@Rx."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_quaternion(m):
    """Rotation matrix -> quaternion (x, y, z, w), scipy/MISO ordering."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def import_monomer_engine():
    """Import MISO's monomer_building module (adds it to sys.path once)."""
    repo = Path(__file__).resolve().parents[3]
    mb_dir = repo / "MISO" / "src" / "monomer_building"
    if not (mb_dir / "monomer_building.py").exists():
        raise ImportError(f"monomer_building.py not found under {mb_dir}")
    if str(mb_dir) not in sys.path:
        sys.path.insert(0, str(mb_dir))
    import monomer_building  # noqa: E402
    return monomer_building


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

# Ordered head-group atoms per linkage: (main_chain_keys[], branch (key, main_index)).
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


class PositionMonomerDialog(QtWidgets.QDialog):
    """Build MISO monomers from SMILES and place them on the STM image.

    Workflow:
      1. Fill the table: SMILES, ring type (chair/boat/...), anomer (alpha/beta)
         and number of copies for each monomer.
      2. *Build monomers* runs monomer_building.generate_monomer_conformers with
         the chosen puckering/anomer filter and keeps the lowest-energy match,
         then instantiates the requested copies.
      3. Select an instance, click *Place* to drop its centre on the image, and
         use RX/RY/RZ to rotate it in 3D (projected top-down onto the image).
      4. *Export CSV* writes the COM (Angstrom) plus the placement rotation
         matrix and quaternion for every instance, for downstream MISO analysis.
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.setWindowTitle("Position monomer")
        self.setMinimumSize(1080, 700)
        self.resize(1200, 780)

        self._updating = False
        self._place_cid = None
        self._templates = {}      # row index -> template dict
        self._instances = []      # list of instance dicts
        self._overlay_artists = []

        # Lipids: point-defined chains (start/end/optional mid + carbons + linkage).
        self._lipids = []         # parallel to lipid table rows
        self._lipid_engine = None
        self._lipid_cid = None
        self._lipid_stage = 0     # 0=start, 1=end, 2=mid

        loaded = load_demo_image(viewer)
        self._img = loaded["img"]
        self._px = loaded["px"]
        self._scan_dir = loaded["scan_dir"]
        self._sxm_path = loaded["sxm_path"]
        self._load_error = loaded["error"]

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Two columns: image on the left, all controls on the right (scrollable)
        # so the growing set of tools never overflows the dialog.
        root = QtWidgets.QHBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        loaded = self._img is not None
        if loaded:
            info = (f"{self._sxm_path.name}  |  {self._img.shape[1]}×{self._img.shape[0]} px"
                    f"  |  {self._px[0]*10:.3f} Å/px  |  scan_dir={self._scan_dir or '?'}")
        else:
            info = f"⚠ Image not loaded: {self._load_error}"

        # ---- left column: info + canvas ----
        left = QtWidgets.QVBoxLayout()
        self.info_lbl = QtWidgets.QLabel(info)
        self.info_lbl.setWordWrap(True)
        left.addWidget(self.info_lbl)
        if loaded:
            self._build_canvas(left)
        else:
            left.addStretch()
        root.addLayout(left, stretch=3)

        # ---- right column: controls inside a scroll area ----
        panel = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(panel)
        col.setSpacing(8)
        col.setContentsMargins(0, 0, 0, 0)

        # --- include options ---
        opt_row = QtWidgets.QHBoxLayout()
        opt_row.addWidget(QtWidgets.QLabel("Sugars always available."))
        self.aa_chk = QtWidgets.QCheckBox("Include amino acids")
        self.aa_chk.toggled.connect(self._on_include_aa_toggled)
        opt_row.addWidget(self.aa_chk)
        self.lipid_chk = QtWidgets.QCheckBox("Include lipids")
        self.lipid_chk.setEnabled(loaded)
        self.lipid_chk.toggled.connect(self._on_include_lipids_toggled)
        opt_row.addWidget(self.lipid_chk)
        opt_row.addStretch()
        col.addLayout(opt_row)

        # --- subunit table (sugars + amino acids can be mixed) ---
        col.addWidget(QtWidgets.QLabel(
            "Subunits to build (mix for glycopeptides):"))
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Type", "Name", "SMILES / Residue", "Ring type", "Anomer", "Copies"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        for c in (0, 1, 3, 4, 5):
            hdr.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setFixedHeight(150)
        col.addWidget(self.table)

        def_row = QtWidgets.QHBoxLayout()
        self.add_sugar_btn = QtWidgets.QPushButton("Add sugar")
        self.add_sugar_btn.clicked.connect(lambda: self._add_table_row("Sugar"))
        self.add_aa_btn = QtWidgets.QPushButton("Add amino acid")
        self.add_aa_btn.setEnabled(False)
        self.add_aa_btn.clicked.connect(lambda: self._add_table_row("Amino acid"))
        del_btn = QtWidgets.QPushButton("Remove selected")
        del_btn.clicked.connect(self._remove_table_row)
        self.build_btn = QtWidgets.QPushButton("Build")
        self.build_btn.setEnabled(loaded)
        self.build_btn.clicked.connect(self._build_monomers)
        def_row.addWidget(self.add_sugar_btn)
        def_row.addWidget(self.add_aa_btn)
        def_row.addWidget(del_btn)
        def_row.addStretch()
        def_row.addWidget(self.build_btn)
        col.addLayout(def_row)
        self._add_table_row("Sugar")

        # --- instance controls ---
        col.addWidget(QtWidgets.QLabel("Placed subunits (select to move/rotate):"))
        ctl = QtWidgets.QHBoxLayout()
        self.inst_list = QtWidgets.QListWidget()
        self.inst_list.setMinimumWidth(200)
        self.inst_list.setFixedHeight(150)
        self.inst_list.currentRowChanged.connect(self._on_instance_selected)
        ctl.addWidget(self.inst_list)

        form_box = QtWidgets.QVBoxLayout()
        self.place_btn = QtWidgets.QPushButton("Place: OFF")
        self.place_btn.setCheckable(True)
        self.place_btn.setEnabled(loaded)
        self.place_btn.toggled.connect(self._on_place_toggled)
        form_box.addWidget(self.place_btn)

        grid = QtWidgets.QGridLayout()
        self.spin = {}
        specs = [("com_x", "COM X (Å)", -1e5, 1e5, 0.5),
                 ("com_y", "COM Y (Å)", -1e5, 1e5, 0.5),
                 ("rx", "Rotate X (°)", -180, 180, 5.0),
                 ("ry", "Rotate Y (°)", -180, 180, 5.0),
                 ("rz", "Rotate Z (°)", -180, 180, 5.0)]
        for i, (key, label, lo, hi, step) in enumerate(specs):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(2)
            sb.valueChanged.connect(self._on_spin_changed)
            grid.addWidget(QtWidgets.QLabel(label), i, 0)
            grid.addWidget(sb, i, 1)
            self.spin[key] = sb
        form_box.addLayout(grid)
        form_box.addStretch()
        ctl.addLayout(form_box)
        col.addLayout(ctl)

        # --- lipids (point-defined chains) ---
        self.lipid_group = QtWidgets.QGroupBox("Lipids")
        self.lipid_group.setVisible(False)
        lg = QtWidgets.QVBoxLayout(self.lipid_group)
        self.lipid_table = QtWidgets.QTableWidget(0, 5)
        self.lipid_table.setHorizontalHeaderLabels(
            ["Start (Å)", "End (Å)", "Mid (Å)", "Carbons (0=auto)", "Linkage"])
        lhdr = self.lipid_table.horizontalHeader()
        for c in range(3):
            lhdr.setSectionResizeMode(c, QtWidgets.QHeaderView.Stretch)
        for c in (3, 4):
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
        lbtn.addWidget(add_lip)
        lbtn.addWidget(rm_lip)
        lbtn.addWidget(self.lipid_pick_btn)
        lbtn.addWidget(clr_lip)
        lbtn.addStretch()
        lg.addLayout(lbtn)
        self.lipid_status = QtWidgets.QLabel("Add a lipid, then Pick points → click Start, End, (Mid).")
        self.lipid_status.setWordWrap(True)
        lg.addWidget(self.lipid_status)
        col.addWidget(self.lipid_group)

        # --- export ---
        csv_row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel("Export CSV:")
        csv_row.addWidget(lbl)
        self.csv_le = QtWidgets.QLineEdit()
        self.csv_le.setPlaceholderText("monomers.csv")
        csv_row.addWidget(self.csv_le)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse_csv)
        csv_row.addWidget(browse)
        col.addLayout(csv_row)

        self.export_btn = QtWidgets.QPushButton("Export CSV")
        self.export_btn.setEnabled(loaded)
        self.export_btn.clicked.connect(self._export_csv)
        col.addWidget(self.export_btn)
        col.addStretch()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(440)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        root.addWidget(scroll, stretch=2)

        self._update_default_csv_name()

    def _build_canvas(self, layout):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self._fig = Figure(figsize=(5, 5))
        self._ax = self._fig.add_subplot(111)
        self._ax.imshow(self._img, cmap="magma", origin="lower", interpolation="nearest")
        self._ax.set_xlabel("col (px)")
        self._ax.set_ylabel("row (px)")
        self._fig.tight_layout()
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(340)
        layout.addWidget(NavigationToolbar(self._canvas, self))
        layout.addWidget(self._canvas, stretch=1)

    def _on_include_aa_toggled(self, checked):
        self.add_aa_btn.setEnabled(bool(checked))

    # ------------------------------------------------------------------ lipids
    def _on_include_lipids_toggled(self, checked):
        self.lipid_group.setVisible(bool(checked))
        if not checked and self.lipid_pick_btn.isChecked():
            self.lipid_pick_btn.setChecked(False)

    def _add_lipid_row(self):
        r = self.lipid_table.rowCount()
        self.lipid_table.insertRow(r)
        for c in range(3):
            it = QtWidgets.QTableWidgetItem("—")
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
            "atoms": None, "atom_types": None, "bonds": None,
        })
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
        self._redraw_overlay()

    def _on_lipid_selected(self, row):
        self._redraw_overlay()

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
        self._redraw_overlay()

    def _on_lipid_pick_toggled(self, active):
        canvas = getattr(self, "_canvas", None)
        if canvas is None:
            self.lipid_pick_btn.setChecked(False)
            return
        if active:
            # Only one canvas-pick mode at a time.
            if self.place_btn.isChecked():
                self.place_btn.setChecked(False)
            if self.lipid_table.currentRow() < 0 and self._lipids:
                self.lipid_table.setCurrentCell(0, 0)
            self._lipid_stage = 0
            self._lipid_cid = canvas.mpl_connect("button_press_event", self._on_lipid_canvas_click)
            self.lipid_pick_btn.setText("Pick points: ON")
            self._update_lipid_status()
        else:
            if self._lipid_cid is not None:
                canvas.mpl_disconnect(self._lipid_cid)
                self._lipid_cid = None
            self.lipid_pick_btn.setText("Pick points: OFF")

    def _update_lipid_status(self):
        stage_name = ["Start", "End", "Mid (optional)"][min(self._lipid_stage, 2)]
        r = self.lipid_table.currentRow()
        who = self._lipids[r]["label"] if 0 <= r < len(self._lipids) else "?"
        self.lipid_status.setText(f"{who}: next click sets {stage_name}.")

    def _clear_lipid_points(self):
        r = self.lipid_table.currentRow()
        if not (0 <= r < len(self._lipids)):
            return
        lip = self._lipids[r]
        lip["start"] = lip["end"] = lip["mid"] = None
        lip["atoms"] = lip["atom_types"] = lip["bonds"] = None
        for c in range(3):
            self.lipid_table.item(r, c).setText("—")
        self._lipid_stage = 0
        self._update_lipid_status()
        self._redraw_overlay()

    def _on_lipid_canvas_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        r = self.lipid_table.currentRow()
        if not (0 <= r < len(self._lipids)):
            QtWidgets.QMessageBox.information(self, "No lipid", "Select a lipid row first.")
            return
        x_ang, y_ang, _h = self._pixel_to_ang(event.xdata, event.ydata)
        lip = self._lipids[r]
        key = ("start", "end", "mid")[min(self._lipid_stage, 2)]
        lip[key] = [x_ang, y_ang]
        self.lipid_table.item(r, min(self._lipid_stage, 2)).setText(f"{x_ang:.2f}, {y_ang:.2f}")
        self._lipid_stage = min(self._lipid_stage + 1, 3)
        self._rebuild_lipid_geometry(lip)
        self._update_lipid_status()
        self._redraw_overlay()

    def _rebuild_lipid_geometry(self, lip):
        """Build head-group + zig-zag backbone via MISO lipid_building."""
        lip["atoms"] = lip["atom_types"] = lip["bonds"] = None
        if lip["start"] is None or lip["end"] is None:
            return
        if self._lipid_engine is None:
            try:
                self._lipid_engine = import_lipid_engine()
            except Exception as exc:
                self.lipid_status.setText(f"⚠ lipid engine import failed: {exc}")
                return
        L = self._lipid_engine
        start = np.array([lip["start"][0], lip["start"][1], 0.0])
        end = np.array([lip["end"][0], lip["end"][1], 0.0])
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
            bb = L.build_lipid_backbone(cur, end, mid, use_fabrik=True, n_carbons=n_carbons)
            chain = [np.asarray(c, dtype=float) for c in bb["chain_carbons"]]
        except Exception as exc:
            self.lipid_status.setText(f"⚠ lipid build failed: {exc}")
            return

        first_c_idx = len(atoms)
        for i, c in enumerate(chain):
            atoms.append(c)
            atom_types.append("C")
            if i > 0:
                bonds.append((first_c_idx + i - 1, first_c_idx + i))
        if main_last_idx is not None and chain:
            bonds.append((main_last_idx, first_c_idx))   # head-group → first carbon

        lip["atoms"] = np.array(atoms)
        lip["atom_types"] = atom_types
        lip["bonds"] = bonds

    def _add_table_row(self, kind="Sugar", text=""):
        """Add a subunit row. kind is 'Sugar' or 'Amino acid'."""
        r = self.table.rowCount()
        self.table.insertRow(r)

        # Column 0: read-only Type tag (drives how the row is interpreted).
        type_item = QtWidgets.QTableWidgetItem(kind)
        type_item.setFlags(type_item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.table.setItem(r, 0, type_item)

        # Column 1: MISO name. For sugars this is the key used in the YAML
        # (sugars:/experimental_positions:), so it must be user-set (e.g. GlcN).
        # For amino acids it is filled from the residue code at build time.
        name_item = QtWidgets.QTableWidgetItem("")
        if kind == "Sugar":
            name_item.setToolTip("MISO monomer name used in the YAML, e.g. GlcN, Gal, KDO")
        else:
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
            name_item.setText("—")
        self.table.setItem(r, 1, name_item)

        # Column 2: SMILES (sugar) or residue code (amino acid).
        item = QtWidgets.QTableWidgetItem(text)
        if kind == "Amino acid":
            item.setToolTip("1- or 3-letter residue code, e.g. N / Asn")
        self.table.setItem(r, 2, item)

        # Columns 3-4: ring type / anomer — only meaningful for sugars.
        if kind == "Sugar":
            ring = QtWidgets.QComboBox()
            ring.addItems(RING_TYPES)
            self.table.setCellWidget(r, 3, ring)
            anom = QtWidgets.QComboBox()
            anom.addItems(ANOMERS)
            self.table.setCellWidget(r, 4, anom)
        else:
            for c in (3, 4):
                dash = QtWidgets.QTableWidgetItem("—")
                dash.setFlags(dash.flags() & ~QtCore.Qt.ItemIsEditable)
                dash.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(r, c, dash)

        # Column 5: copies.
        copies = QtWidgets.QSpinBox()
        copies.setRange(1, 999)
        copies.setValue(1)
        self.table.setCellWidget(r, 5, copies)

    def _remove_table_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    # ------------------------------------------------------------------ build
    def _collect_defs(self):
        """Read the table into a list of per-row dicts describing each subunit."""
        defs = []
        for r in range(self.table.rowCount()):
            type_item = self.table.item(r, 0)
            kind = (type_item.text().strip() if type_item else "Sugar")
            name_item = self.table.item(r, 1)
            name_text = (name_item.text().strip() if name_item else "")
            text_item = self.table.item(r, 2)
            text = (text_item.text().strip() if text_item else "")
            if not text:
                continue
            copies = self.table.cellWidget(r, 5).value()
            if kind == "Sugar":
                ring = self.table.cellWidget(r, 3).currentText()
                anom = self.table.cellWidget(r, 4).currentText()
                # Name is the MISO key; fall back to an auto name if left blank.
                name = name_text or f"Sugar{r+1}"
                defs.append({"row": r, "kind": "sugar", "name": name, "smiles": text,
                             "ring": ring, "anomer": anom, "copies": copies,
                             "conf_name_hint": None,
                             "desc": f"{name}: {text} ({ring}/{anom})"})
            else:
                name, smiles = resolve_aa(text)
                defs.append({"row": r, "kind": "aa", "name": name, "smiles": smiles,
                             "ring": "n/a", "anomer": "n/a", "copies": copies,
                             "conf_name_hint": name,
                             "desc": name or f"'{text}' (unknown residue)"})
        return defs

    def _build_monomers(self):
        defs = self._collect_defs()
        if not defs:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to build", "Add at least one subunit first.")
            return

        engine = None
        if any(d["kind"] == "sugar" for d in defs):
            try:
                engine = import_monomer_engine()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "MISO engine", f"Could not import monomer builder:\n{exc}")
                return

        self._templates.clear()
        self._instances.clear()
        self.inst_list.clear()

        # Default placement: image centre in Angstrom.
        H, W = self._img.shape
        cx = (W / 2.0) * self._px[0] * 10.0
        cy = (H / 2.0) * self._px[1] * 10.0

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        failures = []
        try:
            for d in defs:
                r = d["row"]
                if d["smiles"] is None:
                    failures.append(f"row {r+1}: {d['desc']}")
                    continue
                if d["kind"] == "sugar":
                    tmpl = self._build_sugar_template(engine, d["smiles"], d["ring"], d["anomer"])
                else:
                    tmpl = self._build_aa_template(d["smiles"], d["conf_name_hint"])
                    # Reflect the resolved residue name back into the Name column.
                    nm = self.table.item(r, 1)
                    if nm is not None:
                        nm.setText(d["name"] or "")
                if tmpl is None:
                    failures.append(f"row {r+1}: {d['desc']}")
                    continue
                self._templates[r] = tmpl
                for c in range(d["copies"]):
                    self._instances.append({
                        "label": f"M{r+1}.{c+1}",
                        "name": d["name"], "kind": d["kind"],
                        "smiles": d["smiles"], "ring": d["ring"], "anomer": d["anomer"],
                        "conf_name": tmpl["conf_name"],
                        "rel": tmpl["rel"], "atom_types": tmpl["atom_types"],
                        "bonds": tmpl["bonds"],
                        "rigid": tmpl.get("rigid"),      # exact MISO monomer data (sugars)
                        "com": [cx, cy, 0.0], "euler": [0.0, 0.0, 0.0],
                    })
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        for inst in self._instances:
            self.inst_list.addItem(f"{inst['label']}  [{inst['conf_name'].split('_rank')[0]}]")
        self._redraw_overlay()
        if self._instances:
            self.inst_list.setCurrentRow(0)

        msg = f"Built {len(self._instances)} instance(s) from {len(self._templates)} subunit(s)."
        if failures:
            msg += "\n\nCould not build (no matching conformer / unknown residue):\n  " + "\n  ".join(failures)
        QtWidgets.QMessageBox.information(self, "Build", msg)

    @staticmethod
    def _template_from_mol(mol, conf_name):
        """Heavy-atom coordinates/bonds/COM from an embedded RDKit mol."""
        from rdkit.Chem import RemoveHs
        mol = RemoveHs(mol)                       # H are added at the very end
        conf = mol.GetConformer()
        coords, atom_types, masses = [], [], []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
            atom_types.append(atom.GetSymbol())
            masses.append(atom.GetMass())
        coords = np.asarray(coords, dtype=float)
        com = np.average(coords, weights=np.asarray(masses), axis=0)
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        return {"conf_name": conf_name, "rel": coords - com,
                "atom_types": atom_types, "bonds": bonds}

    def _build_sugar_template(self, engine, smiles, ring, anom):
        known_ring = None if ring == "Any" else ring
        known_anom = None if anom == "Any" else anom
        try:
            confs = engine.generate_monomer_conformers(
                smiles, known_ring_type=known_ring, known_anomer=known_anom)
        except Exception as exc:
            print(f"generate_monomer_conformers failed for {smiles}: {exc}")
            return None
        if not confs:
            return None
        # Conformers are returned lowest-energy first (rank1).
        conf_name = next(iter(confs))
        data = confs[conf_name]
        coords = np.asarray(data["coordinates"], dtype=float)
        com = np.asarray(data["COM"], dtype=float)
        atom_types = list(data["atom_types"])
        bonds = []
        mol = data.get("molecule")
        if mol is not None:
            for b in mol.GetBonds():
                bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        # Capture the EXACT rigid monomer data MISO would otherwise regenerate
        # (relative_coordinates, atom_types, carbon_map, oh_map, anomer,
        # anomeric_oxygen_idx, quaternion) for this single chosen conformer, so a
        # fixed-orientation run reuses this geometry instead of re-embedding.
        rigid = None
        try:
            rigid = engine.extract_rigid_monomer_data({conf_name: data})
        except Exception as exc:
            print(f"extract_rigid_monomer_data failed for {smiles}: {exc}")
        return {"conf_name": conf_name, "rel": coords - com,
                "atom_types": atom_types, "bonds": bonds, "rigid": rigid}

    def _build_aa_template(self, smiles, name):
        """Single lowest-energy conformer for an amino acid (embed + MMFF)."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                return None
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            return self._template_from_mol(mol, name)
        except Exception as exc:
            print(f"amino-acid build failed for {name} ({smiles}): {exc}")
            return None

    # ------------------------------------------------------------------ geometry
    def _ang_to_pixel(self, x_ang, y_ang):
        H, W = self._img.shape
        col = x_ang / (self._px[0] * 10.0)
        orig_y = y_ang / (self._px[1] * 10.0)
        row = (orig_y - 1.0) if self._scan_dir == "down" else (H - 1.0 - orig_y)
        return col, row

    def _pixel_to_ang(self, col, row):
        H, W = self._img.shape
        col = int(max(0, min(W - 1, round(col))))
        row = int(max(0, min(H - 1, round(row))))
        orig_x = col
        orig_y = (row + 1) if self._scan_dir == "down" else (H - 1 - row)
        oy = min(max(orig_y, 0), H - 1)
        height = float(self._img[oy, orig_x])
        x_ang = orig_x * self._px[0] * 10.0
        y_ang = orig_y * self._px[1] * 10.0
        return x_ang, y_ang, height

    def _instance_abs_coords(self, inst):
        """Rotate the monomer and translate to its COM (Angstrom)."""
        Rm = euler_to_matrix(*inst["euler"])
        rotated = inst["rel"] @ Rm.T
        return rotated + np.asarray(inst["com"], dtype=float)

    # ------------------------------------------------------------------ selection
    def _active_instance(self):
        r = self.inst_list.currentRow()
        if 0 <= r < len(self._instances):
            return self._instances[r]
        return None

    def _on_instance_selected(self, _row):
        inst = self._active_instance()
        if inst is None:
            return
        self._updating = True
        self.spin["com_x"].setValue(inst["com"][0])
        self.spin["com_y"].setValue(inst["com"][1])
        self.spin["rx"].setValue(inst["euler"][0])
        self.spin["ry"].setValue(inst["euler"][1])
        self.spin["rz"].setValue(inst["euler"][2])
        self._updating = False
        self._redraw_overlay()

    def _on_spin_changed(self, _val):
        if self._updating:
            return
        inst = self._active_instance()
        if inst is None:
            return
        inst["com"][0] = self.spin["com_x"].value()
        inst["com"][1] = self.spin["com_y"].value()
        inst["euler"] = [self.spin["rx"].value(), self.spin["ry"].value(),
                         self.spin["rz"].value()]
        self._redraw_overlay()

    # ------------------------------------------------------------------ placing
    def _on_place_toggled(self, active):
        canvas = getattr(self, "_canvas", None)
        if canvas is None:
            self.place_btn.setChecked(False)
            return
        if active:
            if getattr(self, "lipid_pick_btn", None) is not None and self.lipid_pick_btn.isChecked():
                self.lipid_pick_btn.setChecked(False)
            self._place_cid = canvas.mpl_connect("button_press_event", self._on_canvas_click)
            self.place_btn.setText("Place: ON (click image)")
        else:
            if self._place_cid is not None:
                canvas.mpl_disconnect(self._place_cid)
                self._place_cid = None
            self.place_btn.setText("Place: OFF")

    def _on_canvas_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        inst = self._active_instance()
        if inst is None:
            QtWidgets.QMessageBox.information(self, "No instance", "Select a monomer instance first.")
            return
        x_ang, y_ang, height = self._pixel_to_ang(event.xdata, event.ydata)
        inst["com"] = [x_ang, y_ang, height]
        self._updating = True
        self.spin["com_x"].setValue(x_ang)
        self.spin["com_y"].setValue(y_ang)
        self._updating = False
        self._redraw_overlay()

    # ------------------------------------------------------------------ drawing
    def _clear_overlay(self):
        for art in self._overlay_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._overlay_artists = []

    def _redraw_overlay(self):
        ax = getattr(self, "_ax", None)
        canvas = getattr(self, "_canvas", None)
        if ax is None or canvas is None:
            return
        self._clear_overlay()
        active = self._active_instance()
        for inst in self._instances:
            abs_c = self._instance_abs_coords(inst)
            cols, rows, colors, sizes = [], [], [], []
            px = {}
            for i, (xyz, sym) in enumerate(zip(abs_c, inst["atom_types"])):
                c, rw = self._ang_to_pixel(xyz[0], xyz[1])
                px[i] = (c, rw)
                cols.append(c)
                rows.append(rw)
                colors.append(ATOM_COLORS.get(sym, "#8e24aa"))
                sizes.append(ATOM_SIZE.get(sym, 22))
            is_active = inst is active
            for (a, b) in inst["bonds"]:
                (ca, ra), (cb, rb) = px[a], px[b]
                ln, = ax.plot([ca, cb], [ra, rb], "-",
                              color="#4fc3f7" if is_active else "#90a4ae",
                              lw=1.6 if is_active else 1.0, zorder=18, alpha=0.9)
                self._overlay_artists.append(ln)
            if cols:
                sc = ax.scatter(cols, rows, c=colors, s=sizes,
                                edgecolors="white", linewidths=0.4,
                                zorder=20, alpha=0.95)
                self._overlay_artists.append(sc)
            # COM marker + label
            comc, comr = self._ang_to_pixel(inst["com"][0], inst["com"][1])
            star, = ax.plot([comc], [comr], marker="*",
                            color="#ffd600" if is_active else "#ffab00",
                            ms=13 if is_active else 10, mec="black", mew=0.6, zorder=22)
            lbl = ax.annotate(inst["label"], xy=(comc, comr), xytext=(5, 5),
                              textcoords="offset points", color="#ffee00",
                              fontsize=8, zorder=23)
            self._overlay_artists.extend([star, lbl])
        self._draw_lipids(ax)
        canvas.draw_idle()

    def _draw_lipids(self, ax):
        sel = self.lipid_table.currentRow() if hasattr(self, "lipid_table") else -1
        for idx, lip in enumerate(self._lipids):
            is_sel = idx == sel
            # start / end / mid markers
            for key, color in (("start", "#43a047"), ("end", "#e53935"), ("mid", "#fb8c00")):
                p = lip.get(key)
                if p is None:
                    continue
                c, rw = self._ang_to_pixel(p[0], p[1])
                m, = ax.plot([c], [rw], marker="s" if key != "mid" else "D",
                             color=color, ms=8 if is_sel else 6,
                             mec="white", mew=0.7, zorder=24)
                self._overlay_artists.append(m)
                if key == "start":
                    t = ax.annotate(lip["label"], xy=(c, rw), xytext=(5, 5),
                                    textcoords="offset points", color="#c8e6c9",
                                    fontsize=8, zorder=25)
                    self._overlay_artists.append(t)
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
                self._overlay_artists.append(ln)
            cols = [px[i][0] for i in range(len(atoms))]
            rows = [px[i][1] for i in range(len(atoms))]
            colors = [ATOM_COLORS.get(s, "#8e24aa") for s in lip["atom_types"]]
            sc = ax.scatter(cols, rows, c=colors, s=18,
                            edgecolors="white", linewidths=0.3, zorder=19, alpha=0.95)
            self._overlay_artists.append(sc)

    # ------------------------------------------------------------------ export
    def _browse_csv(self):
        default_name = self.csv_le.text().strip() or "monomers.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", default_name, "CSV Files (*.csv);;All Files (*)")
        if path:
            self.csv_le.setText(path)

    def _export_csv(self):
        import csv
        placed_lipids = [l for l in self._lipids if l.get("atoms") is not None]
        if not self._instances and not placed_lipids:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to export", "Build/place subunits or lipids first.")
            return
        out_path = self.csv_le.text().strip() or "monomers.csv"
        written = []

        if self._instances:
            header = ["Instance", "SMILES", "RingType", "Anomer", "Conformer",
                      "COM_X (Angstrom)", "COM_Y (Angstrom)", "COM_Z (Angstrom)",
                      "Rot_X (deg)", "Rot_Y (deg)", "Rot_Z (deg)",
                      "quat_x", "quat_y", "quat_z", "quat_w",
                      "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"]
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                for inst in self._instances:
                    Rm = euler_to_matrix(*inst["euler"])
                    q = matrix_to_quaternion(Rm)
                    row = [inst["label"], inst["smiles"], inst["ring"], inst["anomer"],
                           inst["conf_name"],
                           f"{inst['com'][0]:.6f}", f"{inst['com'][1]:.6f}", f"{inst['com'][2]:.6f}",
                           f"{inst['euler'][0]:.4f}", f"{inst['euler'][1]:.4f}", f"{inst['euler'][2]:.4f}",
                           f"{q[0]:.8f}", f"{q[1]:.8f}", f"{q[2]:.8f}", f"{q[3]:.8f}"]
                    row += [f"{v:.8f}" for v in Rm.flatten()]
                    w.writerow(row)
            written.append(Path(out_path).name)
            written += [Path(p).name for p in self._export_miso_inputs(out_path)]
            written += [Path(p).name for p in self._export_monomer_pickle(out_path)]

        if placed_lipids:
            written += [Path(p).name for p in self._export_lipids_csv(out_path, placed_lipids)]

        self._export_png(out_path)
        written.append(Path(out_path).with_suffix(".png").name)
        QtWidgets.QMessageBox.information(
            self, "Done",
            f"Saved {len(self._instances)} subunit(s) and {len(placed_lipids)} lipid(s):\n  "
            + "\n  ".join(written)
            + "\n\nMISO fixed-orientation run — in the YAML set:\n"
              "  circle_input_path: *_positions.csv\n"
              "  orientation_csv_path: *_orientations.csv\n"
              "  monomer_data_path: *_monomer_data.pkl   (reuse exact geometry)\n"
              "  use_fixed_orientation: true\n"
              "then write your connectivity, referencing the point indices (row order) "
              "and the sugar Names.")

    def _export_miso_inputs(self, out_path):
        """Write MISO-ready positions (circle_input) + orientations (by point index).

        <stem>_positions.csv    -> drop-in ``circle_input_path``; one row per placed
                                   subunit, point index = row order (the value used
                                   in ``experimental_positions``).
        <stem>_orientations.csv -> ``orientation_csv_path``; ``Point`` + row-major
                                   3x3 rotation R00..R22 (+quaternion) consumed by
                                   MISO's fixed-orientation mode. Columns match
                                   pipeline.load_orientations.
        """
        import csv
        stem = Path(out_path).with_suffix("")
        pos_path = f"{stem}_positions.csv"
        ori_path = f"{stem}_orientations.csv"

        with open(pos_path, "w", newline="") as f:
            w = csv.writer(f)
            # Name/Instance are extra reference columns; MISO's load_circle_data
            # reads only the "(Angstrom)"/Height columns, so they are ignored there.
            w.writerow(["Point", "Name", "Instance", "Original_X", "Original_Y",
                        "X (Angstrom)", "Y (Angstrom)", "Height", "Z (Angstrom)"])
            for i, inst in enumerate(self._instances):
                cx, cy, cz = inst["com"][0], inst["com"][1], inst["com"][2]
                col, row = self._ang_to_pixel(cx, cy)
                w.writerow([i, inst.get("name", ""), inst["label"],
                            int(round(col)), int(round(row)),
                            f"{cx:.6f}", f"{cy:.6f}", f"{cz:.6f}", 0.0])

        with open(ori_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Point", "Instance", "Type", "Conformer",
                        "quat_x", "quat_y", "quat_z", "quat_w",
                        "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"])
            for i, inst in enumerate(self._instances):
                Rm = euler_to_matrix(*inst["euler"])
                q = matrix_to_quaternion(Rm)
                kind = "aa" if inst["ring"] == "n/a" else "sugar"
                row = [i, inst["label"], kind, inst["conf_name"],
                       f"{q[0]:.8f}", f"{q[1]:.8f}", f"{q[2]:.8f}", f"{q[3]:.8f}"]
                row += [f"{v:.8f}" for v in Rm.flatten()]
                w.writerow(row)
        return [pos_path, ori_path]

    def _export_monomer_pickle(self, out_path):
        """Pickle exact per-sugar rigid monomer data so MISO reuses this geometry.

        Produces {name: rigid_body_data} matching pipeline.extract_monomer_data
        output (each rigid_body_data is {conformer_name: {COM, relative_coordinates,
        atom_types, quaternion, carbon_map, oh_map, anomer, anomeric_oxygen_idx}}).
        MISO loads this via monomer_data_path and skips conformer regeneration, so
        the placed geometry is reproduced exactly. Amino acids are omitted (MISO
        builds those from the peptide sequence, not from this table).
        """
        import pickle
        monomer_data = {}
        for inst in self._instances:
            if inst.get("kind") != "sugar" or inst.get("rigid") is None:
                continue
            # Instances of the same Name share geometry; first one wins.
            monomer_data.setdefault(inst["name"], inst["rigid"])
        if not monomer_data:
            return []
        pkl_path = f"{Path(out_path).with_suffix('')}_monomer_data.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(monomer_data, f)
        return [pkl_path]

    def _export_lipids_csv(self, out_path, placed_lipids):
        """Write MISO-shaped lipid params + a companion per-atom coordinate file."""
        import csv
        stem = Path(out_path).with_suffix("")
        params_path = f"{stem}_lipids.csv"
        atoms_path = f"{stem}_lipids_atoms.csv"

        def fmt(p):
            return (f"{p[0]:.6f}", f"{p[1]:.6f}") if p else ("", "")

        with open(params_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Lipid", "Start_X", "Start_Y", "End_X", "End_Y",
                        "Mid_X", "Mid_Y", "Carbons", "Linkage", "N_built_atoms"])
            for lip in placed_lipids:
                sx, sy = fmt(lip["start"])
                ex, ey = fmt(lip["end"])
                mx, my = fmt(lip["mid"])
                carbons = lip["carbons"] if lip["carbons"] else "auto"
                w.writerow([lip["label"], sx, sy, ex, ey, mx, my,
                            carbons, lip["linkage"], len(lip["atoms"])])

        with open(atoms_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Lipid", "Atom_Index", "Element",
                        "X (Angstrom)", "Y (Angstrom)", "Z (Angstrom)"])
            for lip in placed_lipids:
                for i, (xyz, sym) in enumerate(zip(lip["atoms"], lip["atom_types"])):
                    w.writerow([lip["label"], i, sym,
                                f"{xyz[0]:.6f}", f"{xyz[1]:.6f}", f"{xyz[2]:.6f}"])
        return [params_path, atoms_path]

    def _export_png(self, csv_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if self._img is None:
            return
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(self._img, cmap="magma", origin="lower", interpolation="nearest")
        for inst in self._instances:
            abs_c = self._instance_abs_coords(inst)
            px = {i: self._ang_to_pixel(xyz[0], xyz[1]) for i, xyz in enumerate(abs_c)}
            for (a, b) in inst["bonds"]:
                (ca, ra), (cb, rb) = px[a], px[b]
                ax.plot([ca, cb], [ra, rb], "-", color="#90a4ae", lw=0.9, zorder=18)
            for i, sym in enumerate(inst["atom_types"]):
                c, rw = px[i]
                ax.plot(c, rw, marker="o", ms=3.5,
                        color=ATOM_COLORS.get(sym, "#8e24aa"),
                        mec="white", mew=0.3, zorder=20)
            comc, comr = self._ang_to_pixel(inst["com"][0], inst["com"][1])
            ax.plot(comc, comr, marker="*", color="#ffd600", ms=10, mec="black", mew=0.5, zorder=22)
            ax.annotate(inst["label"], xy=(comc, comr), xytext=(4, 4),
                        textcoords="offset points", color="#ffee00", fontsize=7, zorder=23)
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
                        color=ATOM_COLORS.get(sym, "#8e24aa"), mec="white", mew=0.3, zorder=19)
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
            self.csv_le.setText(f"{stem}_monomers.csv")

    def closeEvent(self, event):
        canvas = getattr(self, "_canvas", None)
        if canvas is not None:
            if self._place_cid is not None:
                canvas.mpl_disconnect(self._place_cid)
            if self._lipid_cid is not None:
                canvas.mpl_disconnect(self._lipid_cid)
        super().closeEvent(event)
