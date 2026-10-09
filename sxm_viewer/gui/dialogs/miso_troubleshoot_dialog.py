"""Explain-only assistant for a failed MISO run."""
from __future__ import annotations

from ..._shared import QtCore, QtWidgets
from ...utils.miso_assistant import AssistantError
from ...utils.miso_troubleshoot import TOOL_DESCRIPTIONS, FailureReport, request_explanation
from .miso_assistant_dialog import ServiceSettingsPanel

DEFAULT_QUESTION = "Why did this MISO run fail?"


class _ExplainSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(str, str)


class _ExplainWorker(QtCore.QRunnable):
    def __init__(self, config, key, report, history):
        super().__init__()
        self.config, self.key = config, key
        self.report, self.history = report, history
        self.signals = _ExplainSignals()

    def run(self):
        try:
            reply = request_explanation(self.config, self.key, self.report, self.history)
        except AssistantError as exc:
            self.signals.finished.emit("", str(exc))
        else:
            self.signals.finished.emit(reply, "")
        finally:
            self.key = ""


class MISOTroubleshootDialog(QtWidgets.QDialog):
    """Shows offline checks and, optionally, asks a hosted LLM to explain the failure.

    The assistant only explains; it never edits the YAML, the CSV or any setting.
    """

    def __init__(self, report: FailureReport, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Why did MISO fail?")
        self.resize(820, 720)
        self.report = report
        self._history = []
        self._consent_config = None
        self._worker = None
        self._closed = False
        self.finished.connect(self._on_closed)
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            f"<b>MISO stopped with exit code {report.exit_code}.</b><br>"
            "This assistant only explains what went wrong. It never changes your YAML, "
            "CSV, or settings.")
        header.setWordWrap(True)
        layout.addWidget(header)

        checks_box = QtWidgets.QGroupBox("Offline checks (nothing sent)")
        checks_layout = QtWidgets.QVBoxLayout(checks_box)
        self.checks_list = QtWidgets.QListWidget()
        self.checks_list.addItems(report.findings or ["No inconsistencies found by the local checks."])
        self.checks_list.setMaximumHeight(110)
        checks_layout.addWidget(self.checks_list)
        layout.addWidget(checks_box)

        online_box = QtWidgets.QGroupBox("Ask an LLM to explain (optional, needs internet)")
        online_layout = QtWidgets.QVBoxLayout(online_box)
        self.service = ServiceSettingsPanel(self)
        online_layout.addWidget(self.service)
        preview_btn = QtWidgets.QPushButton("Show exactly what will be sent")
        preview_btn.clicked.connect(self._show_preview)
        online_layout.addWidget(preview_btn)
        self.transcript = QtWidgets.QPlainTextEdit()
        self.transcript.setReadOnly(True)
        online_layout.addWidget(self.transcript, 1)
        row = QtWidgets.QHBoxLayout()
        self.question_edit = QtWidgets.QLineEdit()
        self.question_edit.setPlaceholderText(DEFAULT_QUESTION)
        self.question_edit.returnPressed.connect(self._send)
        self.send_btn = QtWidgets.QPushButton("Explain the failure")
        self.send_btn.clicked.connect(self._send)
        row.addWidget(self.question_edit, 1)
        row.addWidget(self.send_btn)
        online_layout.addLayout(row)
        self.status = QtWidgets.QLabel("Nothing has been sent.")
        self.status.setWordWrap(True)
        online_layout.addWidget(self.status)
        layout.addWidget(online_box, 1)

        self.service.status.connect(self.status.setText)
        self.service.changed.connect(self._reset_consent)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        layout.addWidget(close)
        self.service.load_settings()

    def _on_closed(self, _result):
        self._closed = True
        self.service.key_edit.clear()

    def _reset_consent(self):
        self._consent_config = None

    def _show_preview(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Data that will be sent")
        dialog.resize(760, 600)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel(
            "Only this text (plus your questions) is sent. Paths and user names are removed."))
        text = QtWidgets.QPlainTextEdit(self.report.preview())
        text.setReadOnly(True)
        layout.addWidget(text)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec_()

    def _ask_consent(self, config) -> bool:
        if self._consent_config == config:
            return True
        parts = "\n".join(f"- {text}" for text in TOOL_DESCRIPTIONS.values())
        answer = QtWidgets.QMessageBox.question(
            self, "Send failure details?",
            f"Send to {config.endpoint} using model '{config.model}'?\n\n"
            f"The model may read:\n{parts}\n\nplus your questions. The YAML structure includes "
            "sugar SMILES. Click 'Show exactly what will be sent' to review it first. "
            "Provider retention policies and charges apply.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            self.status.setText("Nothing sent. The offline checks above remain available.")
            return False
        self._consent_config = config
        return True

    def _send(self):
        if self._worker is not None:
            return
        question = self.question_edit.text().strip() or DEFAULT_QUESTION
        try:
            config = self.service.ready_config()
        except AssistantError as exc:
            QtWidgets.QMessageBox.warning(self, "Why did MISO fail?", str(exc))
            return
        if not self._ask_consent(config):
            return
        self._history.append({"role": "user", "content": question})
        self.transcript.appendPlainText("You: " + question)
        self.question_edit.clear()
        self.send_btn.setEnabled(False)
        self.status.setText("Waiting for the service (up to a minute)...")
        worker = _ExplainWorker(config, self.service.api_key(), self.report, list(self._history))
        worker.signals.finished.connect(self._on_reply)
        self._worker = worker
        QtCore.QThreadPool.globalInstance().start(worker)

    def _on_reply(self, text, error):
        self._worker = None
        if self._closed:
            return
        self.send_btn.setEnabled(True)
        if error:
            self._history.pop()
            self.status.setText(error)
            self.transcript.appendPlainText("(not answered: " + error + ")")
            return
        self._history.append({"role": "assistant", "content": text})
        self.transcript.appendPlainText("Assistant: " + text.strip() + "\n")
        self.status.setText("Explanation received. Nothing was changed.")
