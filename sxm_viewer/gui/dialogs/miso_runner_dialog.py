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
        self._append(f"[MISO] Results dir: {results_dir}")
        self._append(f"[MISO] Config:      {yaml_path}")
        self._append(f"[MISO] CSV:         {csv_path}")
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
                Chem.MolToMol2File(mol, str(results_dir / f"{stem}.mol2"))
                converted += 1
            except Exception as e:
                self._append(f"[MISO] Could not convert {sdf_path.name}: {e}")
        if converted:
            self._append(f"[MISO] Converted {converted} SDF → mol + mol2")

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
