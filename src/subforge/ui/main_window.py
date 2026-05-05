import logging
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

from subforge.config import (
    DEFAULT_URL,
    WHISPER_MODEL,
    OUTPUT_DIR,
)
from subforge.pipeline.processor import SubtitlePipeline, PipelineCancelled
from subforge.transcription.faster_whisper_transcriber import (
    SUPPORTED_MODELS as WHISPER_MODELS,
)
from subforge.translation.factory import BACKEND_NAMES


class PipelineWorker(QObject):
    progress = Signal(str, str)
    finished = Signal(str, str)  # srt_path, video_path (or empty)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        url: str,
        model_name: str,
        use_demucs: bool,
        download_mp4: bool = False,
        video_quality: str = "1080p",
        translate_method: str | None = None,
        force: bool = False,
    ):
        super().__init__()
        self.url = url
        self.model_name = model_name
        self.use_demucs = use_demucs
        self.download_mp4 = download_mp4
        self.video_quality = video_quality
        self.translate_method = translate_method
        self.force = force
        self._pipeline: SubtitlePipeline | None = None

    def cancel(self):
        """Request pipeline cancellation (thread-safe: sets an atomic bool)."""
        if self._pipeline is not None:
            self._pipeline.cancel()

    @Slot()
    def run(self):
        try:
            self._pipeline = SubtitlePipeline(
                url=self.url,
                model_name=self.model_name,
                output_dir=OUTPUT_DIR,
                use_demucs=self.use_demucs,
                download_mp4=self.download_mp4,
                video_quality=self.video_quality,
                translate_method=self.translate_method,
                progress_callback=lambda step, detail: self.progress.emit(step, detail),
                force=self.force,
            )
            srt_path = self._pipeline.run()
            video_path = (
                str(self._pipeline.video_path) if self._pipeline.video_path else ""
            )
            self.finished.emit(str(srt_path), video_path)
        except PipelineCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SubForge")
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
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel)
        url_row.addWidget(self.cancel_button)
        layout.addLayout(url_row)

        # Settings row
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_MODELS)
        whisper_idx = self.model_combo.findText(WHISPER_MODEL)
        if whisper_idx >= 0:
            self.model_combo.setCurrentIndex(whisper_idx)
        settings_row.addWidget(self.model_combo)

        self.demucs_check = QCheckBox("Use Demucs")
        self.demucs_check.setChecked(True)
        settings_row.addWidget(self.demucs_check)

        self.download_mp4_check = QCheckBox("Download MP4")
        self.download_mp4_check.setChecked(False)
        self.download_mp4_check.setToolTip(
            "Download video (with audio) for testing with SRT"
        )
        settings_row.addWidget(self.download_mp4_check)

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["480p", "720p", "1080p", "1440p", "2160p"])
        self.quality_combo.setCurrentText("1080p")
        self.quality_combo.setToolTip("Video quality")
        settings_row.addWidget(self.quality_combo)

        settings_row.addWidget(QLabel("Translate:"))
        self.translate_combo = QComboBox()
        self.translate_combo.addItems(["none"] + BACKEND_NAMES)
        self.translate_combo.setCurrentText("none")
        self.translate_combo.setToolTip("Translation backend (none = disabled)")
        settings_row.addWidget(self.translate_combo)

        self.force_check = QCheckBox("Force Re-run")
        self.force_check.setChecked(False)
        self.force_check.setToolTip("Ignore all cached results and re-run from scratch")
        settings_row.addWidget(self.force_check)

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

    def _append_log(self, line: str):
        self.log.appendPlainText(line)

    @Slot()
    def _on_start(self):
        if self._thread is not None:
            return

        url = self.url_input.text().strip() or DEFAULT_URL
        model_name = self.model_combo.currentText()

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.open_folder_button.setEnabled(False)
        self.result_label.setText("Running…")
        self.log.clear()
        self._append_log(f"Starting: {url}")
        self._append_log(f"Model: {model_name}")

        translate_text = self.translate_combo.currentText()
        self._thread = QThread(self)
        self._worker = PipelineWorker(
            url=url,
            model_name=model_name,
            use_demucs=self.demucs_check.isChecked(),
            download_mp4=self.download_mp4_check.isChecked(),
            video_quality=self.quality_combo.currentText(),
            translate_method=None if translate_text == "none" else translate_text,
            force=self.force_check.isChecked(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.finished.connect(lambda *_: self._thread.quit())
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    @Slot(str, str)
    def _on_progress(self, step: str, detail: str):
        if detail:
            self._append_log(f"[{step}] {detail}")
        else:
            self._append_log(f"[{step}]")

    @Slot(str, str)
    def _on_finished(self, srt_path: str, video_path: str):
        self._last_srt = Path(srt_path)
        result_text = f"SRT: {srt_path}"
        if video_path:
            result_text += f"\nMP4: {video_path}"
            self._append_log(f"Video: {video_path}")
        self.result_label.setText(result_text)
        self.open_folder_button.setEnabled(True)
        self._append_log(f"Done: {srt_path}")

    @Slot(str)
    def _on_failed(self, message: str):
        self.result_label.setText("Failed.")
        self._append_log(f"ERROR: {message}")
        QMessageBox.critical(self, "Pipeline failed", message)

    @Slot()
    def _on_cancel(self):
        """User clicked Cancel — request graceful stop."""
        if self._worker is not None:
            self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self._append_log("Cancelling… (will stop after current step)")

    @Slot()
    def _on_cancelled(self):
        """Pipeline confirmed cancellation."""
        self.result_label.setText("Cancelled.")
        self._append_log("Pipeline cancelled by user.")

    @Slot()
    def _cleanup_thread(self):
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    @Slot()
    def _on_open_folder(self):
        if self._last_srt is None:
            return
        folder = self._last_srt.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def run():
    from subforge.config import LOG_LEVEL
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s - %(name)s - %(message)s")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
