import logging
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot, QUrl, QMimeData, Qt
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from subforge.config import (
    DEFAULT_URL,
    WHISPER_MODEL,
    OUTPUT_DIR,
    TARGET_LANGUAGES,
    TRANSLATE_TGT_LANG,
    ASR_BACKEND,
    ASR_SOURCE_LANGUAGE,
)
from subforge.pipeline.processor import SubtitlePipeline, PipelineCancelled
from subforge.transcription.faster_whisper_transcriber import (
    SUPPORTED_MODELS as WHISPER_MODELS,
)
from subforge.transcription.funasr_transcriber import SUPPORTED_MODELS as FUNASR_MODELS
from subforge.transcription.factory import BACKEND_NAMES as ASR_BACKEND_NAMES
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
        use_punctuation: bool = True,
        download_mp4: bool = False,
        video_quality: str = "1080p",
        translate_method: str | None = None,
        target_lang: str = TRANSLATE_TGT_LANG,
        force: bool = False,
        local_file: str | None = None,
        asr_backend: str = ASR_BACKEND,
        source_language: str = ASR_SOURCE_LANGUAGE,
    ):
        super().__init__()
        self.url = url
        self.model_name = model_name
        self.use_demucs = use_demucs
        self.use_punctuation = use_punctuation
        self.download_mp4 = download_mp4
        self.video_quality = video_quality
        self.translate_method = translate_method
        self.target_lang = target_lang
        self.force = force
        self.local_file = local_file
        self.asr_backend = asr_backend
        self.source_language = source_language
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
                use_punctuation=self.use_punctuation,
                download_mp4=self.download_mp4,
                video_quality=self.video_quality,
                translate_method=self.translate_method,
                target_lang=self.target_lang,
                progress_callback=lambda step, detail: self.progress.emit(step, detail),
                force=self.force,
                local_file=self.local_file,
                asr_backend=self.asr_backend,
                source_language=self.source_language,
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

    _SUPPORTED_EXTENSIONS = (
        ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".wma", ".aac",
        ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts", ".m4v",
    )

    _SOURCE_YOUTUBE = 0
    _SOURCE_LOCAL = 1

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SubForge")
        self.resize(720, 560)
        self.setAcceptDrops(True)

        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._last_srt: Path | None = None
        self._local_file: str | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Source selector row ---
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["YouTube URL", "Local File"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_row.addWidget(self.source_combo)
        source_row.addStretch(1)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._on_start)
        source_row.addWidget(self.start_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel)
        source_row.addWidget(self.cancel_button)
        layout.addLayout(source_row)

        # --- Stacked input area ---
        self.input_stack = QStackedWidget()

        # Page 0: YouTube URL
        url_page = QWidget()
        url_lay = QHBoxLayout(url_page)
        url_lay.setContentsMargins(0, 0, 0, 0)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText(DEFAULT_URL)
        url_lay.addWidget(self.url_input, 1)
        self.input_stack.addWidget(url_page)

        # Page 1: Local file
        file_page = QWidget()
        file_lay = QHBoxLayout(file_page)
        file_lay.setContentsMargins(0, 0, 0, 0)
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText(
            "Drop audio/video here, or click Browse…"
        )
        self.file_input.setReadOnly(True)
        file_lay.addWidget(self.file_input, 1)
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self._on_browse)
        file_lay.addWidget(self.browse_button)
        self.input_stack.addWidget(file_page)

        layout.addWidget(self.input_stack)

        # Settings row
        settings_row = QHBoxLayout()

        settings_row.addWidget(QLabel("ASR:"))
        self.asr_backend_combo = QComboBox()
        self.asr_backend_combo.addItems(ASR_BACKEND_NAMES)
        self.asr_backend_combo.setCurrentText(ASR_BACKEND)
        self.asr_backend_combo.currentTextChanged.connect(self._on_asr_backend_changed)
        settings_row.addWidget(self.asr_backend_combo)

        self.source_lang_input = QLineEdit()
        self.source_lang_input.setPlaceholderText("src lang (auto)")
        self.source_lang_input.setFixedWidth(90)
        self.source_lang_input.setToolTip(
            "Source language ISO 639-1 code (e.g. zh). Leave blank for auto-detect."
        )
        settings_row.addWidget(self.source_lang_input)

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

        self.punctuation_check = QCheckBox("Punctuation")
        self.punctuation_check.setChecked(True)
        self.punctuation_check.setToolTip("Restore punctuation using local model")
        settings_row.addWidget(self.punctuation_check)

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
        self.translate_combo.currentTextChanged.connect(self._on_translate_changed)
        settings_row.addWidget(self.translate_combo)

        self.target_lang_label = QLabel("→")
        settings_row.addWidget(self.target_lang_label)
        self.target_lang_combo = QComboBox()
        for code, name in TARGET_LANGUAGES.items():
            self.target_lang_combo.addItem(f"{name} ({code})", userData=code)
        # Default to TRANSLATE_TGT_LANG from config
        default_idx = self.target_lang_combo.findData(TRANSLATE_TGT_LANG)
        if default_idx >= 0:
            self.target_lang_combo.setCurrentIndex(default_idx)
        self.target_lang_combo.setToolTip("Target language for translation")
        settings_row.addWidget(self.target_lang_combo)
        # Initially hidden when translate is "none"
        self.target_lang_label.setVisible(False)
        self.target_lang_combo.setVisible(False)

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

    @Slot(str)
    def _on_asr_backend_changed(self, backend: str):
        """Swap model combo items when ASR backend changes."""
        self.model_combo.clear()
        if backend == "funasr":
            self.model_combo.addItems(FUNASR_MODELS)
        else:
            self.model_combo.addItems(WHISPER_MODELS)
            idx = self.model_combo.findText(WHISPER_MODEL)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

    @Slot(str)
    def _on_translate_changed(self, text: str):
        """Show/hide target language dropdown based on translation backend."""
        enabled = text != "none"
        self.target_lang_label.setVisible(enabled)
        self.target_lang_combo.setVisible(enabled)

    # ------------------------------------------------------------------
    # Source mode switching
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_source_changed(self, index: int):
        """Switch between YouTube URL and Local File input pages."""
        self.input_stack.setCurrentIndex(index)
        # Reset local file state when switching away from local mode
        if index == self._SOURCE_YOUTUBE:
            self._local_file = None
            self.file_input.clear()
            self.file_input.setToolTip("")

    # ------------------------------------------------------------------
    # Local file input helpers
    # ------------------------------------------------------------------

    @Slot()
    def _on_browse(self):
        """Open a file dialog for selecting a local audio/video file."""
        ext_list = " ".join(f"*{e}" for e in self._SUPPORTED_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio or Video File",
            "",
            f"Media Files ({ext_list});;All Files (*)",
        )
        if path:
            self._set_local_file(path)

    def _set_local_file(self, path: str):
        """Set a local file path from browse dialog or drag-and-drop."""
        p = Path(path)
        if p.suffix.lower() not in self._SUPPORTED_EXTENSIONS:
            QMessageBox.warning(
                self,
                "Unsupported File",
                f"File type '{p.suffix}' is not supported.\n\n"
                f"Supported formats:\n{', '.join(self._SUPPORTED_EXTENSIONS)}",
            )
            return
        self._local_file = str(p)
        self.file_input.setText(p.name)
        self.file_input.setToolTip(str(p))
        # Auto-switch to local mode if user was on YouTube page
        if self.source_combo.currentIndex() != self._SOURCE_LOCAL:
            self.source_combo.setCurrentIndex(self._SOURCE_LOCAL)

    # ------------------------------------------------------------------
    # Drag & Drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime: QMimeData = event.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    suffix = Path(url.toLocalFile()).suffix.lower()
                    if suffix in self._SUPPORTED_EXTENSIONS:
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        mime: QMimeData = event.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    suffix = Path(file_path).suffix.lower()
                    if suffix in self._SUPPORTED_EXTENSIONS:
                        self._set_local_file(file_path)
                        event.acceptProposedAction()
                        return
        event.ignore()

    # ------------------------------------------------------------------
    # Pipeline start
    # ------------------------------------------------------------------

    @Slot()
    def _on_start(self):
        if self._thread is not None:
            return

        is_local = self.source_combo.currentIndex() == self._SOURCE_LOCAL
        local_file: str | None = None
        url = ""

        if is_local:
            local_file = self._local_file
            if not local_file:
                QMessageBox.warning(
                    self, "No file selected",
                    "Please select an audio or video file first.",
                )
                return
        else:
            url = self.url_input.text().strip() or DEFAULT_URL

        model_name = self.model_combo.currentText()
        asr_backend = self.asr_backend_combo.currentText()
        source_language = self.source_lang_input.text().strip() or ASR_SOURCE_LANGUAGE

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.open_folder_button.setEnabled(False)
        self.result_label.setText("Running…")
        self.log.clear()

        if local_file:
            self._append_log(f"Starting: {Path(local_file).name} (local file)")
        else:
            self._append_log(f"Starting: {url}")
        self._append_log(f"ASR: {asr_backend}, Model: {model_name}")

        translate_text = self.translate_combo.currentText()
        target_lang = self.target_lang_combo.currentData() or TRANSLATE_TGT_LANG
        self._thread = QThread(self)
        self._worker = PipelineWorker(
            url=url,
            model_name=model_name,
            use_demucs=self.demucs_check.isChecked(),
            use_punctuation=self.punctuation_check.isChecked(),
            download_mp4=self.download_mp4_check.isChecked(),
            video_quality=self.quality_combo.currentText(),
            translate_method=None if translate_text == "none" else translate_text,
            target_lang=target_lang,
            force=self.force_check.isChecked(),
            local_file=local_file,
            asr_backend=asr_backend,
            source_language=source_language,
        )
        thread = self._thread
        worker = self._worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(lambda *_: thread.quit())
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(self._cleanup_thread)
        thread.start()

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

    logging.basicConfig(
        level=LOG_LEVEL, format="%(levelname)s - %(name)s - %(message)s"
    )
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
