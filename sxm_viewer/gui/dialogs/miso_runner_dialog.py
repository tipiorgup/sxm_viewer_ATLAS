from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path

from ..._shared import QtCore, QtWidgets, QtGui


class MISORunnerDialog(QtWidgets.QDialog):
    """Run the MISO pipeline (module_B.py) with a YAML config + CSV + NPZ."""

    _MISO_DIR = Path(__file__).resolve().parents[3] / "MISO"
    @classmethod
    def _engine_info(cls, use_help_fallback=False):
        """What the installed MISO engine accepts (see utils/miso_engine.py)."""
        from ...utils.miso_engine import inspect_engine
        return inspect_engine(cls._MISO_DIR, use_help_fallback=use_help_fallback)

    @classmethod
    def _supported_flags(cls):
        """Command line options module_B.py declares."""
        return set(cls._engine_info().options)

    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.setWindowTitle("MISO Runner")
        self.setMinimumWidth(620)
        self._process = None
        self._tmp_yaml = None
        self._queue = []
        self._current_job = None
        self._job_total = 0
        self._job_summary = []
        self._failure = None
        self._job_log_start = 0
        self._default_polymers = 5
        self._stopped_by_user = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # Queue of configurations: one row per YAML, each with its own positions
        # CSV and its own number of polymers. Rows run one after the other.
        self.queue_table = QtWidgets.QTableWidget(0, 2)
        self.queue_table.setHorizontalHeaderLabels(["Config YAML", "Polymers"])
        header = self.queue_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.queue_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.queue_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.queue_table.setMinimumHeight(110)
        root.addWidget(self.queue_table)

        queue_btns = QtWidgets.QHBoxLayout()
        self.add_yaml_btn = QtWidgets.QPushButton("Add YAML(s)...")
        self.add_yaml_btn.setToolTip("Select one or several YAML files. They run one after the other.")
        self.add_yaml_btn.clicked.connect(self._browse_yamls)
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_queue)
        for btn in (self.add_yaml_btn, self.remove_btn, self.clear_btn):
            queue_btns.addWidget(btn)
        queue_btns.addStretch()
        root.addLayout(queue_btns)

        # Coordinates are shared by every queued YAML.
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        self.csv_le, csv_row = self._file_row("Browse...", "CSV files (*.csv);;All files (*)")
        form.addRow("Positions CSV (all YAMLs):", csv_row)
        root.addLayout(form)

        self.fixed_ori_chk = QtWidgets.QCheckBox(
            "Use fixed orientation (skip QUEST alignment for placed monomers), applies to all YAMLs")
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
        if "--debug_checkpoints" not in self._supported_flags():
            self.debug_chk.setEnabled(False)
            self.debug_chk.setChecked(False)
            self.debug_chk.setToolTip(
                "This MISO version no longer has the checkpoint option.")
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
        param_row.addWidget(QtWidgets.QLabel("Polymers per YAML:"))
        self.poly_spin = QtWidgets.QSpinBox()
        self.poly_spin.setRange(1, 100)
        self.poly_spin.setValue(5)
        self.poly_spin.setFixedWidth(70)
        self.poly_spin.setToolTip(
            "Default for every queued YAML. Rows you changed by hand keep their own value.")
        self.poly_spin.valueChanged.connect(self._on_default_polymers_changed)
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
        self.why_btn = QtWidgets.QPushButton("Why did MISO fail?")
        self.why_btn.setEnabled(False)
        self.why_btn.setToolTip(
            "Available after a failed run. Shows offline checks and can ask an optional "
            "online LLM to explain the error. It never changes your files.")
        self.why_btn.clicked.connect(self._explain_failure)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.why_btn)
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

    # ------------------------------------------------------------------
    # YAML queue
    # ------------------------------------------------------------------

    def _browse_yamls(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open YAML files", "", "YAML files (*.yml *.yaml);;All files (*)")
        for path in paths:
            self._add_yaml(path)

    def _row_yaml(self, row):
        return self.queue_table.item(row, 0).text()

    def _row_polymers(self, row):
        return self.queue_table.cellWidget(row, 1).value()

    def _read_yaml_fields(self, path):
        """(fixed flag, path values) from a YAML, or None after showing a warning."""
        try:
            import yaml
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self, "Missing dependency", "PyYAML is required to load MISO configuration.")
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            if not isinstance(config, dict):
                raise ValueError("The YAML must contain a configuration mapping.")
            fixed = config.get("use_fixed_orientation", False)
            if not isinstance(fixed, bool):
                raise ValueError("use_fixed_orientation must be true or false.")
            values = {}
            for key in ("circle_input_path", "orientation_csv_path", "monomer_data_path"):
                value = config.get(key)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"{key} must be a file path.")
                values[key] = (str((Path(path).resolve().parent / value).resolve())
                               if value else "")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            QtWidgets.QMessageBox.warning(
                self, "YAML error", f"Could not load configuration:\n{path}\n{exc}")
            return None
        return fixed, values

    def _add_yaml(self, path):
        path = str(Path(path).resolve())
        for row in range(self.queue_table.rowCount()):
            if self._row_yaml(row) == path:
                return False
        fields = self._read_yaml_fields(path)
        if fields is None:
            return False
        fixed, values = fields
        row = self.queue_table.rowCount()
        self.queue_table.insertRow(row)
        yaml_item = QtWidgets.QTableWidgetItem(path)
        yaml_item.setToolTip(path)
        self.queue_table.setItem(row, 0, yaml_item)
        spin = QtWidgets.QSpinBox()
        spin.setRange(1, 100)
        spin.setValue(self.poly_spin.value())
        self.queue_table.setCellWidget(row, 1, spin)
        if self.queue_table.rowCount() == 1:
            # The first YAML seeds the shared fields; later ones only fill gaps.
            self.fixed_ori_chk.setChecked(fixed)
        for edit, key in ((self.csv_le, "circle_input_path"),
                          (self.ori_le, "orientation_csv_path"),
                          (self.monomer_le, "monomer_data_path")):
            if not edit.text().strip() and values[key]:
                edit.setText(values[key])
        return True

    def _remove_selected(self):
        rows = sorted({idx.row() for idx in self.queue_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.queue_table.removeRow(row)

    def _clear_queue(self):
        self.queue_table.setRowCount(0)

    def _on_default_polymers_changed(self, value):
        """Move rows still at the previous default to the new one."""
        previous = self._default_polymers
        for row in range(self.queue_table.rowCount()):
            spin = self.queue_table.cellWidget(row, 1)
            if spin.value() == previous:
                spin.setValue(value)
        self._default_polymers = value

    def _prefill_from_viewer(self):
        """Queue the last exported YAML, or guess CSV/orientation/monomer files."""
        last_yaml = getattr(self.viewer, "last_monomer_yaml", None)
        if last_yaml:
            self._add_yaml(last_yaml)
            return
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
        rows = self.queue_table.rowCount()
        csv_path = self.csv_le.text().strip()
        missing = []
        if rows == 0:
            missing.append("No config YAML selected.")
        for row in range(rows):
            yaml_path = self._row_yaml(row)
            if not Path(yaml_path).is_file():
                missing.append(f"Config YAML not found:\n  {yaml_path}")
        if not csv_path:
            missing.append("Positions CSV not selected.")
        elif not Path(csv_path).is_file():
            missing.append(f"Positions CSV not found:\n  {csv_path}")

        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Missing files",
                "Cannot run MISO \u2014 please fix the following:\n\n" + "\n\n".join(missing)
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

        jobs = []
        used_names = set()
        for row in range(rows):
            yaml_path = self._row_yaml(row)
            try:
                with open(yaml_path, "r") as f:
                    cfg = yaml.safe_load(f)
                if not isinstance(cfg, dict):
                    raise ValueError("The YAML must contain a configuration mapping.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "YAML error", f"Could not read config file:\n{yaml_path}\n{e}")
                return
            self._prepare_config(cfg, yaml_path, csv_path)

            results_dir = Path(csv_path).parent / "results"
            if rows > 1:
                # Output names come from the YAML's sxm_file, so two YAMLs of the
                # same acquisition would overwrite each other in one folder.
                name = Path(yaml_path).stem
                unique, k = name, 2
                while unique in used_names:
                    unique, k = f"{name}_{k}", k + 1
                used_names.add(unique)
                results_dir = results_dir / unique
            results_dir.mkdir(parents=True, exist_ok=True)
            jobs.append({
                "yaml": yaml_path, "csv": csv_path, "cfg": cfg,
                "n_polymers": self._row_polymers(row), "results_dir": results_dir,
            })

        self.log.clear()
        self._queue = jobs
        self._job_total = len(jobs)
        self._job_summary = []
        self._failure = None
        self._stopped_by_user = False
        self.why_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._start_next_job()

    def _prepare_config(self, cfg, yaml_path, csv_path):
        """Apply the dialog values to one loaded YAML mapping, in place."""
        cfg["circle_input_path"] = csv_path
        cfg.pop("stm_grid_path", None)

        # Orientations CSV / fixed-orientation flag. When the
        # checkbox is on take these from the dialog fields (only overriding a
        # field that's actually filled, so a yaml value isn't wiped by an empty
        # one). When off, actively clear orientation_csv_path/use_fixed_orientation
        # instead of leaving a stale value from the yaml, module_B.py loads
        # orientation_csv_path whenever it's present regardless of the
        # use_fixed_orientation flag, so an unchecked box has to mean "not
        # used at all", not just "not overridden here". monomer_data_path is
        # independent (reuse exact placed geometry vs regenerate) and is
        # never touched by this checkbox either way.
        # These dialog values are shared by every queued YAML.
        yaml_dir = Path(yaml_path).resolve().parent
        use_fixed = self.fixed_ori_chk.isChecked()

        if use_fixed:
            cfg["use_fixed_orientation"] = True
            ori_path = self.ori_le.text().strip()
            if ori_path:
                cfg["orientation_csv_path"] = ori_path
            monomer_le_path = self.monomer_le.text().strip()
            if monomer_le_path:
                cfg["monomer_data_path"] = monomer_le_path
            ori_cfg = cfg.get("orientation_csv_path")
            if ori_cfg and not Path(ori_cfg).is_absolute():
                cfg["orientation_csv_path"] = str((yaml_dir / ori_cfg).resolve())
        else:
            cfg["use_fixed_orientation"] = False
            cfg.pop("orientation_csv_path", None)

        # A relative monomer_data_path in the yaml is written by the user
        # relative to the yaml's own folder, but module_B.py runs with its
        # working directory set to results_dir, so resolve it here
        # instead of letting it fail to open once the process starts.
        monomer_data_path = cfg.get("monomer_data_path")
        if monomer_data_path and not Path(monomer_data_path).is_absolute():
            cfg["monomer_data_path"] = str((yaml_dir / monomer_data_path).resolve())

    def _start_next_job(self):
        import yaml

        job = self._queue.pop(0)
        self._current_job = job
        cfg = job["cfg"]
        results_dir = job["results_dir"]
        self._results_dir = results_dir
        index = self._job_total - len(self._queue)

        self._tmp_yaml = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, dir=str(results_dir)
        )
        yaml.dump(cfg, self._tmp_yaml)
        self._tmp_yaml.flush()
        self._tmp_yaml.close()

        if index > 1:
            self._append("")
        self._job_log_start = len(self.log.toPlainText())
        if self._job_total > 1:
            self._append("=" * 60)
            self._append(f"[MISO] Job {index}/{self._job_total}: {Path(job['yaml']).name}")
        self._append(f"[MISO] Results dir:        {results_dir}")
        self._append(f"[MISO] Config:             {job['yaml']}")
        self._append(f"[MISO] Polymers:           {job['n_polymers']}")
        self._append(f"[MISO] circle_input_path:  {cfg.get('circle_input_path')}")
        self._append(f"[MISO] orientation_csv_path: {cfg.get('orientation_csv_path')}")
        self._append(f"[MISO] use_fixed_orientation: {cfg.get('use_fixed_orientation', False)}")
        self._append(f"[MISO] monomer_data_path:  {cfg.get('monomer_data_path')}")
        self._append("-" * 60)

        from ...utils.miso_engine import EngineError, build_arguments, compatibility_notes
        info = self._engine_info(use_help_fallback=True)
        try:
            engine_args, notes = build_arguments(
                info, self._tmp_yaml.name, self.iter_spin.value(), job["n_polymers"],
                phase1_kicks=True, debug_checkpoints=self.debug_chk.isChecked())
        except EngineError as exc:
            # The engine itself is unusable, so no later job could work either.
            self._append(f"[MISO] {exc}")
            QtWidgets.QMessageBox.warning(self, "MISO engine", str(exc))
            self._cleanup_tmp()
            self._queue = []
            self._finish_queue()
            return
        notes += compatibility_notes(info, cfg)
        for note in notes:
            self._append(f"[MISO] Note: {note}")
        supported = set(info.options)
        args = ["-X", "utf8", str(self._MISO_DIR / "module_B.py"), *engine_args]

        self._process = QtCore.QProcess(self)
        self._process.setWorkingDirectory(str(results_dir))
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)

        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        existing_pp = env.value("PYTHONPATH", "")
        miso_str = str(self._MISO_DIR)
        env.insert("PYTHONPATH", f"{miso_str}{os.pathsep}{existing_pp}" if existing_pp else miso_str)
        env.insert("DEFAULT_COMPRESSION_STEPS", str(self.comp_spin.value()))
        env.insert("DEFAULT_GRAVITY", str(self.gravity_spin.value()))
        self._process.setProcessEnvironment(env)

        job["settings"] = {
            "iterations": self.iter_spin.value(),
            "n_polymers": job["n_polymers"],
            "compression_steps": self.comp_spin.value(),
            "gravity": self.gravity_spin.value(),
            "use_fixed_orientation": bool(cfg.get("use_fixed_orientation")),
            "debug_checkpoints": ("--debug_checkpoints" in supported
                                  and self.debug_chk.isChecked()),
        }
        from ...utils.miso_engine import console_python
        self._process.start(console_python(), args)

    def _stop(self):
        if self._process and self._process.state() != QtCore.QProcess.NotRunning:
            self._stopped_by_user = True
            self._queue = []
            self._process.kill()
            self._append("[MISO] Stopped by user.")

    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data().decode(errors="replace")
        self._append(data.rstrip())

    def _on_stderr(self):
        data = self._process.readAllStandardError().data().decode(errors="replace")
        self._append(data.rstrip())

    def _on_finished(self, exit_code, exit_status):
        job = self._current_job
        self._append("-" * 60)
        self._append(f"[MISO] Finished (exit code {exit_code})")
        self._cleanup_tmp()
        failed = exit_code != 0 or exit_status == QtCore.QProcess.CrashExit
        status = "stopped" if self._stopped_by_user else ("FAILED" if failed else "ok")
        self._job_summary.append((Path(job["yaml"]).name, job["n_polymers"], status))
        if failed and not self._stopped_by_user:
            if self._failure is None:
                # The troubleshooter explains the first failure of the batch,
                # using only the log lines of that job.
                self._failure = {
                    "log": self.log.toPlainText()[self._job_log_start:],
                    "cfg": dict(job["cfg"]),
                    "settings": job.get("settings", {}),
                    "exit_code": exit_code or 1,
                }
            self._append("[MISO] Click 'Why did MISO fail?' for checks and an explanation.")
        if exit_code == 0:
            self._convert_sdf_outputs()

        if self._queue and not self._stopped_by_user:
            self._start_next_job()
            return
        self._finish_queue()

    def _finish_queue(self):
        if self._job_total > 1 and self._job_summary:
            self._append("=" * 60)
            self._append("[MISO] Batch summary:")
            for name, n_poly, status in self._job_summary:
                self._append(f"[MISO]   {name}  ({n_poly} polymers)  {status}")
        self._current_job = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.why_btn.setEnabled(self._failure is not None and not self._stopped_by_user)

    def _explain_failure(self):
        from ...utils.miso_troubleshoot import build_failure_report
        from .miso_troubleshoot_dialog import MISOTroubleshootDialog

        failure = self._failure
        if not failure:
            return
        report = build_failure_report(failure["log"], failure["cfg"],
                                      failure["settings"], failure["exit_code"])
        MISOTroubleshootDialog(report, self).exec_()

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
