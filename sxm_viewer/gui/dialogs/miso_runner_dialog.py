from __future__ import annotations
import sys
import tempfile
from pathlib import Path

from ..._shared import QtCore, QtWidgets, QtGui


class MISORunnerDialog(QtWidgets.QDialog):
    """Run the MISO pipeline (module_B.py) with a YAML config + CSV + NPZ."""

    _MISO_DIR = Path(__file__).resolve().parents[3] / "MISO"

    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.setWindowTitle("MISO Runner")
        self.setMinimumWidth(620)
        self._process = None
        self._tmp_yaml = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        self.yaml_le, yaml_row = self._file_row("Browse…", "YAML files (*.yml *.yaml);;All files (*)")
        self.csv_le,  csv_row  = self._file_row("Browse…", "CSV files (*.csv);;All files (*)")

        form.addRow("Config YAML:", yaml_row)
        form.addRow("Positions CSV:", csv_row)
        root.addLayout(form)

        self.fixed_ori_chk = QtWidgets.QCheckBox(
            "Use fixed orientation (skip QUEST alignment for placed monomers)")
        self.fixed_ori_chk.toggled.connect(self._on_fixed_ori_toggled)
        root.addWidget(self.fixed_ori_chk)

        # Only relevant once fixed orientation is on, so hidden until checked.
        self._fixed_ori_group = QtWidgets.QWidget()
        fixed_ori_form = QtWidgets.QFormLayout(self._fixed_ori_group)
        fixed_ori_form.setContentsMargins(0, 0, 0, 0)
        fixed_ori_form.setLabelAlignment(QtCore.Qt.AlignRight)
        self.ori_le, ori_row = self._file_row("Browse…", "CSV files (*.csv);;All files (*)")
        self.monomer_le, monomer_row = self._file_row("Browse…", "Pickle files (*.pkl);;All files (*)")
        fixed_ori_form.addRow("Orientations CSV:", ori_row)
        fixed_ori_form.addRow("Monomer data (.pkl, optional):", monomer_row)
        self._fixed_ori_group.setVisible(False)
        root.addWidget(self._fixed_ori_group)

        self.debug_chk = QtWidgets.QCheckBox(
            "Debugging option (save intermediate structures: pre-opt, "
            "post-phase1, post-phase2)")
        root.addWidget(self.debug_chk)

        # Parameters
        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("Iterations:"))
        self.iter_spin = QtWidgets.QSpinBox()
        self.iter_spin.setRange(1, 100_000)
        self.iter_spin.setValue(100)
        self.iter_spin.setFixedWidth(90)
        param_row.addWidget(self.iter_spin)
        param_row.addSpacing(20)
        param_row.addWidget(QtWidgets.QLabel("Polymers:"))
        self.poly_spin = QtWidgets.QSpinBox()
        self.poly_spin.setRange(1, 100)
        self.poly_spin.setValue(5)
        self.poly_spin.setFixedWidth(70)
        param_row.addWidget(self.poly_spin)
        param_row.addSpacing(20)
        param_row.addWidget(QtWidgets.QLabel("Compression steps:"))
        self.comp_spin = QtWidgets.QSpinBox()
        self.comp_spin.setRange(1, 100_000)
        self.comp_spin.setValue(100)
        self.comp_spin.setFixedWidth(90)
        param_row.addWidget(self.comp_spin)
        param_row.addSpacing(20)
        param_row.addWidget(QtWidgets.QLabel("Gravity:"))
        self.gravity_spin = QtWidgets.QDoubleSpinBox()
        self.gravity_spin.setRange(0.01, 100.0)
        self.gravity_spin.setSingleStep(0.5)
        self.gravity_spin.setValue(2.0)
        self.gravity_spin.setFixedWidth(70)
        param_row.addWidget(self.gravity_spin)
        param_row.addStretch()
        root.addLayout(param_row)

        # Run / Stop
        btn_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run MISO")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(280)
        self.log.setFont(QtGui.QFont("Courier New", 9))
        root.addWidget(self.log)

        # Pre-fill CSV/NPZ from last position-coordinates export
        self._prefill_from_viewer()

    def _on_fixed_ori_toggled(self, checked):
        self._fixed_ori_group.setVisible(checked)

    def _file_row(self, label: str, filt: str):
        le = QtWidgets.QLineEdit()
        le.setPlaceholderText("(not selected)")
        btn = QtWidgets.QPushButton(label)
        btn.setFixedWidth(80)
        btn.clicked.connect(lambda: self._browse(le, filt))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(le)
        row.addWidget(btn)
        return le, row

    def _browse(self, line_edit: QtWidgets.QLineEdit, filt: str):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "", filt)
        if path:
            line_edit.setText(path)

    def _prefill_from_viewer(self):
        """Fill CSV/NPZ from the last export of PositionCoordinatesDialog."""
        try:
            canvas = getattr(self.viewer, "preview_canvas", None)
            if canvas and canvas.views:
                stem = Path(canvas.views[0].get("file_name", "")).stem
            else:
                stem = Path(self.viewer.last_preview[0]).stem
            if stem:
                csv_guess = Path(stem + "_positions.csv")
                if csv_guess.exists():
                    self.csv_le.setText(str(csv_guess.resolve()))
                ori_guess = Path(stem + "_orientations.csv")
                if ori_guess.exists():
                    self.ori_le.setText(str(ori_guess.resolve()))
                monomer_guess = Path(stem + "_monomer_data.pkl")
                if monomer_guess.exists():
                    self.monomer_le.setText(str(monomer_guess.resolve()))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------

    def _run(self):
        yaml_path = self.yaml_le.text().strip()
        csv_path  = self.csv_le.text().strip()

        missing = []
        if not yaml_path:
            missing.append("Config YAML not selected.")
        elif not Path(yaml_path).is_file():
            missing.append(f"Config YAML not found:\n  {yaml_path}")

        if not csv_path:
            missing.append("Positions CSV not selected.")
        elif not Path(csv_path).is_file():
            missing.append(f"Positions CSV not found:\n  {csv_path}")

        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Missing files",
                "Cannot run MISO — please fix the following:\n\n" + "\n\n".join(missing)
            )
            return

        try:
            import yaml
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self, "Missing dependency",
                "PyYAML is not installed.\nRun:  pip install pyyaml"
            )
            return

        try:
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "YAML error", f"Could not read config file:\n{e}")
            return

        cfg["circle_input_path"] = csv_path
        cfg.pop("stm_grid_path", None)

        # Orientations CSV / monomer pkl / fixed-orientation flag: these only
        # come from the dialog when the checkbox is on and a field is filled,
        # so a yaml that already has them set (circle_input_path is always
        # overridden above, these are not) is left alone otherwise instead of
        # being silently wiped by a hidden, empty field.
        if self.fixed_ori_chk.isChecked():
            cfg["use_fixed_orientation"] = True
            ori_path = self.ori_le.text().strip()
            if ori_path:
                cfg["orientation_csv_path"] = ori_path
            monomer_le_path = self.monomer_le.text().strip()
            if monomer_le_path:
                cfg["monomer_data_path"] = monomer_le_path

        # A relative monomer_data_path in the yaml is written by the user
        # relative to the yaml's own folder, but module_B.py runs with its
        # working directory set to results_dir below, so resolve it here
        # instead of letting it fail to open once the process starts.
        monomer_data_path = cfg.get("monomer_data_path")
        if monomer_data_path and not Path(monomer_data_path).is_absolute():
            cfg["monomer_data_path"] = str((Path(yaml_path).resolve().parent / monomer_data_path).resolve())

        results_dir = Path(csv_path).parent / "results"
        results_dir.mkdir(exist_ok=True)
        self._results_dir = results_dir

        self._tmp_yaml = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, dir=str(results_dir)
        )
        yaml.dump(cfg, self._tmp_yaml)
        self._tmp_yaml.flush()
        self._tmp_yaml.close()

        self.log.clear()
        self._append(f"[MISO] Results dir:        {results_dir}")
        self._append(f"[MISO] Config:             {yaml_path}")
        self._append(f"[MISO] circle_input_path:  {cfg.get('circle_input_path')}")
        self._append(f"[MISO] orientation_csv_path: {cfg.get('orientation_csv_path')}")
        self._append(f"[MISO] use_fixed_orientation: {cfg.get('use_fixed_orientation', False)}")
        self._append(f"[MISO] monomer_data_path:  {cfg.get('monomer_data_path')}")
        self._append("-" * 60)

        self._process = QtCore.QProcess(self)
        self._process.setWorkingDirectory(str(results_dir))
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)

        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        existing_pp = env.value("PYTHONPATH", "")
        miso_str = str(self._MISO_DIR)
        env.insert("PYTHONPATH", f"{miso_str};{existing_pp}" if existing_pp else miso_str)
        env.insert("DEFAULT_COMPRESSION_STEPS", str(self.comp_spin.value()))
        env.insert("DEFAULT_GRAVITY", str(self.gravity_spin.value()))
        self._process.setProcessEnvironment(env)

        args = [
            "-X", "utf8",
            str(self._MISO_DIR / "module_B.py"),
            "--input_file", self._tmp_yaml.name,
            "--iterations", str(self.iter_spin.value()),
            "--n_polymers", str(self.poly_spin.value()),
            "--phase1_kicks",
        ]
        if self.debug_chk.isChecked():
            args.append("--debug_checkpoints")
        self._process.start(sys.executable, args)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop(self):
        if self._process and self._process.state() != QtCore.QProcess.NotRunning:
            self._process.kill()
            self._append("[MISO] Stopped by user.")

    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data().decode(errors="replace")
        self._append(data.rstrip())

    def _on_stderr(self):
        data = self._process.readAllStandardError().data().decode(errors="replace")
        self._append(data.rstrip())

    def _on_finished(self, exit_code, exit_status):
        self._append("-" * 60)
        self._append(f"[MISO] Finished (exit code {exit_code})")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._cleanup_tmp()
        if exit_code == 0:
            self._convert_sdf_outputs()

    def _convert_sdf_outputs(self):
        results_dir = getattr(self, "_results_dir", None)
        if not results_dir:
            return
        try:
            from rdkit import Chem
        except ImportError:
            self._append("[MISO] RDKit not available — skipping mol/mol2 export.")
            return
        sdf_files = list(results_dir.glob("*.sdf"))
        converted = 0
        for sdf_path in sdf_files:
            try:
                suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
                mol = next((m for m in suppl if m is not None), None)
                if mol is None:
                    continue
                stem = sdf_path.stem
                Chem.MolToMolFile(mol, str(results_dir / f"{stem}.mol"))
                converted += 1
            except Exception as e:
                self._append(f"[MISO] Could not convert {sdf_path.name}: {e}")
        if converted:
            self._append(f"[MISO] Converted {converted} SDF → mol")

    def _cleanup_tmp(self):
        if self._tmp_yaml:
            try:
                Path(self._tmp_yaml.name).unlink(missing_ok=True)
            except Exception:
                pass
            self._tmp_yaml = None

    def _append(self, text: str):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self._stop()
        self._cleanup_tmp()
        super().closeEvent(event)
