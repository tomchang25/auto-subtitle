import sys
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from youtube_subtitle_app.config import (
    DEFAULT_URL,
    DEFAULT_MODEL,
    WHISPER_MODEL,
    ENGINE,
    OUTPUT_DIR,
)
from youtube_subtitle_app.pipeline.processor import SubtitlePipeline
from youtube_subtitle_app.transcription.faster_whisper_transcriber import (
    SUPPORTED_MODELS as WHISPER_MODELS,
)


ENGINES = ("faster-whisper", "parakeet")


class PipelineWorker(QObject):
    progress = Signal(str, str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, url: str, engine: str, model_name: str, use_demucs: bool):
        super().__init__()
        self.url = url
        self.engine = engine
        self.model_name = model_name
        self.use_demucs = use_demucs

    @Slot()
    def run(self):
        try:
            pipeline = SubtitlePipeline(
                url=self.url,
                engine=self.engine,
                model_name=self.model_name,
                output_dir=OUTPUT_DIR,
                use_demucs=self.use_demucs,
                progress_callback=lambda step, detail: self.progress.emit(step, detail),
            )
            srt_path = pipeline.run()
            self.finished.emit(str(srt_path))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Subtitle")
        self.resize(720, 560)

        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._last_srt: Path | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # URL row
        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("YouTube URL:"))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText(DEFAULT_URL)
        url_row.addWidget(self.url_input, 1)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._on_start)
        url_row.addWidget(self.start_button)
        layout.addLayout(url_row)

        # Settings row
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Engine:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(ENGINES)
        idx = self.engine_combo.findText(ENGINE)
        if idx >= 0:
            self.engine_combo.setCurrentIndex(idx)
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        settings_row.addWidget(self.engine_combo)

        self.model_label = QLabel("Model:")
        settings_row.addWidget(self.model_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_MODELS)
        whisper_idx = self.model_combo.findText(WHISPER_MODEL)
        if whisper_idx >= 0:
            self.model_combo.setCurrentIndex(whisper_idx)
        settings_row.addWidget(self.model_combo)

        self.demucs_check = QCheckBox("Use Demucs")
        self.demucs_check.setChecked(True)
        settings_row.addWidget(self.demucs_check)
        settings_row.addStretch(1)
        layout.addLayout(settings_row)

        # Progress log
        layout.addWidget(QLabel("Progress:"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        # Result row
        result_row = QHBoxLayout()
        self.result_label = QLabel("No output yet.")
        self.result_label.setWordWrap(True)
        result_row.addWidget(self.result_label, 1)
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self._on_open_folder)
        result_row.addWidget(self.open_folder_button)
        layout.addLayout(result_row)

        self._on_engine_changed(self.engine_combo.currentText())

    def _on_engine_changed(self, engine: str):
        is_whisper = engine == "faster-whisper"
        self.model_label.setVisible(is_whisper)
        self.model_combo.setVisible(is_whisper)

    def _append_log(self, line: str):
        self.log.appendPlainText(line)

    @Slot()
    def _on_start(self):
        if self._thread is not None:
            return

        url = self.url_input.text().strip() or DEFAULT_URL
        engine = self.engine_combo.currentText()
        if engine == "faster-whisper":
            model_name = self.model_combo.currentText()
        else:
            model_name = DEFAULT_MODEL

        self.start_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        self.result_label.setText("Running…")
        self.log.clear()
        self._append_log(f"Starting: {url}")
        self._append_log(f"Engine: {engine}  Model: {model_name}")

        self._thread = QThread(self)
        self._worker = PipelineWorker(
            url=url,
            engine=engine,
            model_name=model_name,
            use_demucs=self.demucs_check.isChecked(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    @Slot(str, str)
    def _on_progress(self, step: str, detail: str):
        if detail:
            self._append_log(f"[{step}] {detail}")
        else:
            self._append_log(f"[{step}]")

    @Slot(str)
    def _on_finished(self, srt_path: str):
        self._last_srt = Path(srt_path)
        self.result_label.setText(f"SRT: {srt_path}")
        self.open_folder_button.setEnabled(True)
        self._append_log(f"Done: {srt_path}")

    @Slot(str)
    def _on_failed(self, message: str):
        self.result_label.setText("Failed.")
        self._append_log(f"ERROR: {message}")
        QMessageBox.critical(self, "Pipeline failed", message)

    @Slot()
    def _cleanup_thread(self):
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.start_button.setEnabled(True)

    @Slot()
    def _on_open_folder(self):
        if self._last_srt is None:
            return
        folder = self._last_srt.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def run():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
