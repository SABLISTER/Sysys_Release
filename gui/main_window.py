"""
Main application window for the Hypothesis Consensus Analyzer.

Contains the MainWindow class with 13 tabs (Settings tab removed — now a
dialog accessible from File > Settings...) and a File menu bar.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

from PySide6.QtCore import Qt, QSize, QThread, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QMenu,
    QPlainTextEdit,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.theme import COLORS
from gui.original_question import OriginalQuestionState
from gui.pipeline_board import (
    PENDING_SUMMARY,
    PipelineBoardDialog,
    collect_stage_progress,
    empty_stage_progress,
    merge_disk_progress,
)
from gui.project_manager import ProjectManager
from gui.widgets import MetricCard, AlignmentBar, SectionHeader, GradeBadge
from gui.workers import (
    AnalysisWorker,
    FullTextRetrievalWorker,
    TextScalingWorker,
    ProjectBackupWorker,
    PipelineSearchWorker,
    PipelineStageWorker,
)
from core.models import (
    AnalysisSession,
)
from core.config import Config, PROVIDER_MODEL_FIELD, normalize_provider_name
from gui.model_picker import EndpointGetter, ModelFetchButton
from process.custom_endpoint import CustomEndpoint

LLM_PROVIDER_CHOICES = ["ollama", "custom", "openai", "anthropic"]
from core.io import atomic_write_json

logger = logging.getLogger(__name__)


# ── Module-level helpers ──────────────────────────────────────────────────

class PipelineState:
    """String-constant namespace for pipeline state tracking."""
    IDLE = "idle"
    SEARCHING = "searching"
    SCORING = "scoring"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    AUDITING = "auditing"
    SYNTHESIZING = "synthesizing"
    FULLTEXT = "fulltext"
    INTERRATER = "interrater"
    ROBUSTNESS = "robustness"
    VERIFYING = "verifying"
    COMPARING = "comparing"
    DONE = "done"


def _paths_resolve_under_root(raw: str, project_root: Path) -> Path | None:
    """Resolve a user path; relative paths are taken under project_root."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        return p
    except OSError:
        return None


def _make_file_row(
    label: str,
    placeholder: str = "",
    browse_save: bool = False,
    file_filter: str = "All files (*)",
) -> tuple:
    """Create a reusable file-path input row (label + QLineEdit + Browse button)."""
    h = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setMinimumWidth(100)
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    edit.setAccessibleName(label.replace(":", "").strip())
    lbl.setBuddy(edit)
    btn = QPushButton("Browse...")
    btn.setAccessibleName(f"Browse for {label.replace(':', '').strip()}")
    h.addWidget(lbl)
    h.addWidget(edit, 1)
    h.addWidget(btn)
    return h, edit, btn


def _make_llm_backend_row(
    default_provider: str = "ollama",
    default_model: str = "",
    extra_providers: list[str] | None = None,
    endpoint_getter: EndpointGetter | None = None,
) -> tuple:
    """Create a reusable LLM provider/model selector row.

    ``extra_providers`` appends non-LLM backends (e.g. "brt_bert") to the
    dropdown for stages that support them, without polluting every row.
    ``endpoint_getter`` adds a Fetch Models button, enabled while the row's
    provider is "custom".
    """
    h = QHBoxLayout()
    combo = QComboBox()
    combo.addItems(LLM_PROVIDER_CHOICES)
    if extra_providers:
        combo.addItems(extra_providers)
    idx = combo.findText(default_provider)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    model_edit = QLineEdit()
    model_edit.setPlaceholderText(default_model or "model name")
    model_edit.setAccessibleName("LLM model")

    provider_lbl = QLabel("Provider:")
    provider_lbl.setBuddy(combo)
    h.addWidget(provider_lbl)
    h.addWidget(combo)

    model_lbl = QLabel("Model:")
    model_lbl.setBuddy(model_edit)
    h.addWidget(model_lbl)
    h.addWidget(model_edit, 1)

    if endpoint_getter is not None:
        fetch_btn = ModelFetchButton(model_edit, endpoint_getter)
        fetch_btn.setEnabled(combo.currentText() == "custom")
        combo.currentTextChanged.connect(
            lambda text: fetch_btn.setEnabled(text == "custom")
        )
        h.addWidget(fetch_btn)
        setattr(model_edit, "_fetch_button", fetch_btn)

    # Track whether the user has hand-edited these fields. Once customized,
    # programmatic syncs (project open, Settings → Apply) must NOT overwrite
    # them — only user edits or a new preference default should change them.
    # Use user-initiated signals (activated / textEdited) so programmatic
    # setText()/setCurrentIndex() calls do not flip the flag.
    setattr(combo, "_user_customized", False)
    combo.activated.connect(lambda *_: setattr(combo, "_user_customized", True))
    setattr(model_edit, "_user_customized", False)
    model_edit.textEdited.connect(
        lambda *_: setattr(model_edit, "_user_customized", True)
    )
    return h, combo, model_edit


# ── MainWindow ────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    Main application window.

    14 tabs (Search, Score & Filter, Extract & Validate, Audit, Synthesis,
    Fulltext & Interrater, Robustness & Verify, Text Scaling, Evidence Map,
    Results Dashboard, Neuroimaging, Compare Runs, Report & Export, Corpus).

    Settings are accessed via File > Settings... (Ctrl+,) which opens
    SettingsDialog.
    """

    def __init__(self):
        super().__init__()
        from gui.settings_persistence import migrate_legacy_settings
        migrate_legacy_settings()
        self.setWindowTitle("Hypothesis Consensus Analyzer \u2014 Sysys Merged Pipeline")
        self.setMinimumSize(QSize(1280, 850))
        self.resize(1440, 950)

        # ── Search analysis state ──
        self.session = AnalysisSession()
        self.hypothesis_components: list = []
        self.articles: dict = {}
        self._fetch_worker: Optional[QThread] = None
        self._analysis_worker: Optional[AnalysisWorker] = None
        self._ft_worker: Optional[FullTextRetrievalWorker] = None
        self._scaling_worker: Optional[TextScalingWorker] = None
        self._wordfish_model = None
        self._stylometry_profiles: dict = {}
        self._corpus_stats: dict = {}
        self._evidence_table_rows: list[dict] = []
        self._evidence_table_source: str = ""
        self._ft_article_row: dict = {}

        # ── Sysys pipeline state ──
        # Auto-load research_config.yaml if it exists in the project root
        try:
            self.pipeline_config: Config = Config.load_config()
        except Exception:
            self.pipeline_config: Config = Config()
        self._restore_saved_keys()  # layer QSettings keys onto Config
        self.pipeline_state: str = PipelineState.IDLE
        self._pipeline_worker: Optional[PipelineStageWorker] = None
        self._search_worker: Optional[PipelineSearchWorker] = None
        self._pipeline_corpus: list = []
        self._pipeline_board: PipelineBoardDialog | None = None
        self._pipeline_progress: dict[int, dict[str, str]] = empty_stage_progress()
        self._run_all_queue: list[str] = []
        self._run_all_active: bool = False
        self._project_path: Path | None = None
        self._original_question_state = OriginalQuestionState()
        self._backup_worker: ProjectBackupWorker | None = None
        self._pending_auto_backup_reason: str = ""
        self._last_auto_backup_monotonic: float = 0.0
        self._auto_backup_timer = QTimer(self)
        self._auto_backup_timer.setSingleShot(True)
        self._auto_backup_timer.timeout.connect(
            lambda: self._run_auto_backup(reason="scheduled")
        )
        self._sysys_search_runs: int = 0
        self._keyword_analysis_ready: bool = False

        # ── Build UI ──
        self._build_menu_bar()
        self._build_tabs()
        self._build_status_bar()
        self._build_output_console()
        self._build_log_window()
        self._install_log_handler()

        # Populate hypothesis components and sync settings to all tabs.
        # Restore persisted LLM defaults (provider/model/base URLs) and any
        # remembered API keys from QSettings first, so the defaults the user
        # last set in Preferences show up in the stage rows on launch.
        self._populate_components_from_config()
        self._restore_saved_keys()
        self._sync_settings_to_tabs()
        self._update_search_workflow_buttons()

        self._show_welcome_if_first_run()
        self._maybe_prompt_auto_backup("first_launch")
        self._setup_shortcuts()
        self._auto_load_config()

    # ── Menu bar ──────────────────────────────────────────────────────

    def _build_menu_bar(self):
        """Create the File, Edit, View, and Help menus."""
        menu_bar = self.menuBar()

        # ── File menu ──
        file_menu = menu_bar.addMenu("&File")

        new_proj_action = QAction("&New Project...", self)
        new_proj_action.setShortcut(QKeySequence("Ctrl+N"))
        new_proj_action.setStatusTip("Create a new SystS project folder")
        new_proj_action.triggered.connect(self._new_project)
        file_menu.addAction(new_proj_action)

        open_proj_action = QAction("&Open Project...", self)
        open_proj_action.setShortcut(QKeySequence("Ctrl+O"))
        open_proj_action.setStatusTip("Open an existing SystS project folder")
        open_proj_action.triggered.connect(self._open_project)
        file_menu.addAction(open_proj_action)

        self._recent_menu = QMenu("Recent Projects", self)
        file_menu.addMenu(self._recent_menu)
        self._populate_recent_menu()

        save_proj_action = QAction("&Save Project", self)
        save_proj_action.setShortcut(QKeySequence("Ctrl+S"))
        save_proj_action.setStatusTip("Save the current project state")
        save_proj_action.triggered.connect(self._save_project)
        file_menu.addAction(save_proj_action)

        backup_proj_action = QAction("&Backup Project...", self)
        backup_proj_action.setStatusTip("Create a timestamped project backup zip")
        backup_proj_action.triggered.connect(self._backup_project)
        file_menu.addAction(backup_proj_action)

        restore_proj_action = QAction("&Restore Project Backup...", self)
        restore_proj_action.setStatusTip("Restore a SystS project backup")
        restore_proj_action.triggered.connect(self._restore_project_backup)
        file_menu.addAction(restore_proj_action)

        auto_backup_action = QAction("Auto &Backup Settings...", self)
        auto_backup_action.setStatusTip("Configure automatic Drive/local-folder backups")
        auto_backup_action.triggered.connect(
            lambda checked=False: self._configure_auto_backup()
        )
        file_menu.addAction(auto_backup_action)

        file_menu.addSeparator()

        # Log window — opened on demand as a floating window (like the console).
        self._toggle_log_action = QAction("&Log Window", self)
        self._toggle_log_action.setShortcut(QKeySequence("Ctrl+L"))
        self._toggle_log_action.setCheckable(True)
        self._toggle_log_action.setChecked(False)
        self._toggle_log_action.setStatusTip("Show/hide the application log window")
        self._toggle_log_action.triggered.connect(self._toggle_log_window)
        file_menu.addAction(self._toggle_log_action)

        file_menu.addSeparator()

        citation_action = QAction("&Citation Guide...", self)
        citation_action.setStatusTip("Show recommended citations for SystS analyses")
        citation_action.triggered.connect(self._open_citation_guide)
        file_menu.addAction(citation_action)

        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.setStatusTip("Open application settings")
        settings_action.triggered.connect(self._open_settings_dialog)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ── Edit menu ──
        edit_menu = menu_bar.addMenu("&Edit")

        center_output_action = QAction("&Show Output Console in Center", self)
        center_output_action.setShortcut(QKeySequence("Ctrl+Shift+`"))
        center_output_action.setStatusTip(
            "Show the output console and move it to the center of the screen"
        )
        center_output_action.triggered.connect(self._show_output_console_centered)
        edit_menu.addAction(center_output_action)

        # ── View menu ──
        view_menu = menu_bar.addMenu("&View")

        self._toggle_output_action = QAction("&Output Console", self)
        self._toggle_output_action.setShortcut(QKeySequence("Ctrl+`"))
        self._toggle_output_action.setCheckable(True)
        self._toggle_output_action.setChecked(True)
        self._toggle_output_action.setStatusTip("Show/hide the output console")
        self._toggle_output_action.triggered.connect(self._toggle_output_console)
        view_menu.addAction(self._toggle_output_action)

        # ── Help menu ──
        help_menu = menu_bar.addMenu("&Help")

        pipeline_board_action = QAction("&Pipeline Board...", self)
        pipeline_board_action.setShortcut(QKeySequence("Ctrl+P"))
        pipeline_board_action.setStatusTip("Show the pipeline status board")
        pipeline_board_action.triggered.connect(self._show_pipeline_board)
        help_menu.addAction(pipeline_board_action)

    def _open_settings_dialog(self):
        """Open the Settings dialog."""
        from gui.dialogs import SettingsDialog
        dlg = SettingsDialog(self, parent=self)
        dlg.exec()

    def _open_citation_guide(self):
        """Open copyable citation guidance for SystS analyses."""
        from gui.dialogs import CitationGuideDialog
        dlg = CitationGuideDialog(self)
        dlg.exec()

    # ── Status bar ────────────────────────────────────────────────────

    def _build_status_bar(self):
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Ready")

    # ── Output console (dockable) ──────────────────────────────────────

    def _build_output_console(self):
        """Create a dockable output console at the bottom of the window."""
        self._output_dock = QDockWidget("Output", self)
        self._output_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._console_output = QTextEdit()
        self._console_output.setReadOnly(True)
        self._console_output.setAccessibleName("Output console")
        # Bound the document: long pipeline runs append unboundedly otherwise.
        self._console_output.document().setMaximumBlockCount(5000)
        self._console_output.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; "
            "font-size: 14pt; "
            "background-color: #1a1a2e; color: #e0e0e0; "
            f"border: 1px solid {COLORS['border']};"
        )
        self._console_output.setMinimumHeight(80)
        layout.addWidget(self._console_output)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._console_output.clear)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._output_dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._output_dock)
        # Float the output console as a separate window by default
        self._output_dock.setFloating(True)
        self._output_dock.resize(700, 350)
        self._output_dock.show()

        # Sync dock visibility with menu checkbox
        self._output_dock.visibilityChanged.connect(
            lambda visible: self._toggle_output_action.setChecked(visible)
        )

    def _toggle_output_console(self, checked: bool):
        """Show or hide the output console dock."""
        if checked:
            self._output_dock.show()
        else:
            self._output_dock.hide()

    def _show_output_console_centered(self):
        """Show the output console as a floating window centered on-screen."""
        dock = self._output_dock
        dock.setFloating(True)
        dock.showNormal()
        dock.show()

        screen = self.windowHandle().screen() if self.windowHandle() else None
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(max(dock.width(), 700), int(available.width() * 0.9))
            height = min(max(dock.height(), 350), int(available.height() * 0.9))
            x = available.x() + (available.width() - width) // 2
            y = available.y() + (available.height() - height) // 2
            dock.setGeometry(x, y, width, height)

        dock.raise_()
        dock.activateWindow()

    # ── Python logging → GUI bridge ─────────────────────────────────────

    def _install_log_handler(self):
        """Route Python logging from all Sysys modules into the GUI console.

        Uses a QObject signal to safely cross from worker threads to the GUI thread.
        """
        from PySide6.QtCore import QObject, Signal as QSignal

        class _Bridge(QObject):
            log_record = QSignal(str, str)  # (message, level)

        class _GUIHandler(logging.Handler):
            def __init__(self, bridge):
                super().__init__()
                self.bridge = bridge

            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.bridge.log_record.emit(msg, record.levelname)
                except Exception:
                    pass

        self._log_bridge = _Bridge()
        self._log_bridge.log_record.connect(self._on_log_record)

        handler = _GUIHandler(self._log_bridge)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.DEBUG)

        # Attach to the root logger for gui.* and process.* modules
        for name in ("gui", "process", "core", "stages", "acquire"):
            log = logging.getLogger(name)
            log.addHandler(handler)
            log.setLevel(logging.DEBUG)

    def _on_log_record(self, message: str, level: str):
        """Slot receiving log records from worker threads."""
        self._log(message, level)

    # ── Help / Welcome ─────────────────────────────────────────────────

    def _add_help_button(self, tab_widget: QWidget, tab_name: str) -> QPushButton:
        """Insert a small '?' help button at the top-right of a tab's layout."""
        from gui.dialogs import HelpDialog

        btn = QPushButton("?")
        btn.setFixedSize(24, 24)
        btn.setToolTip("Learn about this stage")
        btn.setAccessibleName(f"Help for {tab_name} tab")
        btn.setStyleSheet(
            f"QPushButton {{ font-weight: bold; font-size: 13pt; "
            f"padding: 0px; min-height: 0px; }}"
        )
        btn.clicked.connect(lambda checked, n=tab_name: HelpDialog(n, self).exec())

        help_row = QHBoxLayout()
        help_row.setContentsMargins(0, 0, 0, 0)
        help_row.addStretch()
        help_row.addWidget(btn)

        tab_layout = tab_widget.layout()
        if isinstance(tab_layout, QBoxLayout):
            tab_layout.insertLayout(0, help_row)
        return btn

    def _show_welcome_if_first_run(self):
        """Show the welcome dialog if the user hasn't dismissed it."""
        from PySide6.QtCore import QSettings
        from gui.dialogs import WelcomeDialog

        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        if not settings.value("welcome/shown", False, type=bool):
            WelcomeDialog(self).exec()

    def _maybe_prompt_auto_backup(self, reason: str) -> None:
        """Offer automatic backup on first launch or after creating a project."""
        from PySide6.QtCore import QSettings

        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        if settings.value("backup/enabled", False, type=bool):
            return
        if settings.value("backup/prompt_disabled", False, type=bool):
            return
        if reason == "first_launch" and settings.value(
            "backup/first_launch_prompted", False, type=bool
        ):
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Automatic Project Backup")
        box.setText("Turn on automatic project backup?")
        box.setInformativeText(
            "SystS can keep a lightweight backup in your Google Drive folder. "
            "It copies only changed review files by default and skips large full-text downloads."
        )
        enable_btn = box.addButton("Turn On...", QMessageBox.ButtonRole.AcceptRole)
        not_now_btn = box.addButton("Not Now", QMessageBox.ButtonRole.RejectRole)
        dont_ask_btn = box.addButton("No, Don't Ask Again", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(enable_btn)
        box.exec()

        clicked = box.clickedButton()
        if reason == "first_launch":
            settings.setValue("backup/first_launch_prompted", True)
        if clicked == enable_btn:
            self._configure_auto_backup(default_enabled=True)
        elif clicked == dont_ask_btn:
            settings.setValue("backup/prompt_disabled", True)
        elif clicked == not_now_btn:
            return

    # ── Tab widget ────────────────────────────────────────────────────

    def _build_tabs(self):
        """Build tabs with At-a-Glance panel above (Settings tab removed — now a dialog)."""
        # ── Central widget: vertical container ──
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # ── Feature D: At-a-Glance panel ──
        self._glance_frame = QFrame()
        self._glance_frame.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['bg']}; "
            f"border-top: 2px solid #404040; border-left: 2px solid #404040; "
            f"border-bottom: 2px solid #ffffff; border-right: 2px solid #ffffff; }}"
        )
        glance_outer = QHBoxLayout(self._glance_frame)
        glance_outer.setContentsMargins(6, 4, 6, 4)
        glance_outer.setSpacing(8)

        self._card_corpus = MetricCard("Corpus", "0 articles")
        self._card_relevant = MetricCard("Relevant", "0 papers")
        self._card_grade = MetricCard("GRADE", "—")
        self._card_stage = MetricCard("Stage", "Ready")

        glance_outer.addWidget(self._card_corpus)
        glance_outer.addWidget(self._card_relevant)
        glance_outer.addWidget(self._card_grade)
        glance_outer.addWidget(self._card_stage)
        glance_outer.addStretch()

        self._glance_toggle_btn = QPushButton("▲ Hide")
        self._glance_toggle_btn.setFixedWidth(60)
        self._glance_toggle_btn.clicked.connect(self._toggle_glance_panel)
        glance_outer.addWidget(self._glance_toggle_btn)

        central_layout.addWidget(self._glance_frame)

        # ── Tab widget ──
        self.tabs = QTabWidget()
        central_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)

        # Tab 0: Search
        self.tabs.addTab(self._build_search_tab(), "Search")
        # Tab 1: Score & Filter (also hosts Stage 2c Statistical Extraction in
        # its top-right column, and Human Relevance Calibration below)
        self.tabs.addTab(self._build_score_filter_tab(), "Score && Filter")
        # Tab 2: Extract & Validate
        self.tabs.addTab(self._build_extract_validate_tab(), "Extract && Validate")
        # Tab 3: Audit (Stage 5)
        self.tabs.addTab(self._build_audit_tab(), "Audit")
        # Tab 4: Synthesis (Stage 6 — abstract, hypothesis rationale,
        # source citations, and evidence network get their own roomy tab)
        self.tabs.addTab(self._build_synthesis_tab(), "Synthesis")
        # Tab 5: Fulltext & Interrater
        self.tabs.addTab(self._build_fulltext_interrater_tab(), "Fulltext && Interrater")
        # Tab 6: Robustness & Verify
        self.tabs.addTab(self._build_robustness_verify_tab(), "Robustness && Verify")
        # Tab 7: Text Scaling
        self.tabs.addTab(self._build_scaling_tab(), "Text Scaling")
        # Tab 8: Evidence Map
        self.tabs.addTab(self._build_evidence_tab(), "Evidence Map")
        # Tab 9: Results Dashboard
        self.tabs.addTab(self._build_alignment_tab(), "Results Dashboard")
        # Tab 10: Neuroimaging
        self.tabs.addTab(self._build_neuroimaging_tab(), "Neuroimaging")
        # Tab 11: Compare Runs
        self.tabs.addTab(self._build_compare_runs_tab(), "Compare Runs")
        # Tab 12: Report & Export
        self.tabs.addTab(self._build_report_export_tab(), "Report && Export")
        # Tab 13: Corpus Browser (Feature B)
        self.tabs.addTab(self._build_corpus_tab(), "Corpus")

        # The Log is no longer a tab; it lives in a floating window opened from
        # the File menu (built separately in __init__).

    # ── Tab 0: Search ─────────────────────────────────────────────────

    def _build_search_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Hypothesis component list
        layout.addWidget(SectionHeader("Hypothesis Components",
                                       "Edit components of the research hypothesis"))
        self.component_list = QTableWidget(0, 5)
        self.component_list.setHorizontalHeaderLabels(["ID", "Label", "Keywords", "Weight", "Boolean Term"])
        self.component_list.setMinimumHeight(120)
        layout.addWidget(self.component_list)

        comp_btns = QHBoxLayout()
        edit_btn = QPushButton("Edit Component")
        edit_btn.clicked.connect(self._edit_component)
        reset_btn = QPushButton("Reset Components")
        reset_btn.clicked.connect(self._reset_components)
        comp_btns.addWidget(edit_btn)
        comp_btns.addWidget(reset_btn)
        comp_btns.addStretch()
        layout.addLayout(comp_btns)

        # Sysys search question. This is the first-class input; components
        # can be edited ahead of time or regenerated from this question.
        layout.addWidget(SectionHeader("Search Question"))
        self.sysys_nl_query = QLineEdit()
        self.sysys_nl_query.setPlaceholderText("Natural language research question...")
        self.sysys_nl_query.setAccessibleName("Search question")
        self.sysys_nl_query.textChanged.connect(self._on_search_question_changed)
        layout.addWidget(self.sysys_nl_query)

        self.boolean_query_preview = QTextEdit()
        self.boolean_query_preview.setReadOnly(True)
        self.boolean_query_preview.setAccessibleName("Weighted Boolean query preview")
        self.boolean_query_preview.setMaximumHeight(74)
        self.boolean_query_preview.setPlaceholderText(
            "Weighted Boolean query preview will appear after components are prepared."
        )
        layout.addWidget(self.boolean_query_preview)

        # API source checkboxes
        layout.addWidget(SectionHeader("Data Sources"))
        src_row = QHBoxLayout()
        self.src_pubmed = QCheckBox("PubMed")
        self.src_pubmed.setChecked(True)
        self.src_s2 = QCheckBox("Semantic Scholar")
        self.src_s2.setChecked(True)
        self.src_openalex = QCheckBox("OpenAlex")
        self.src_openalex.setChecked(True)
        self.src_elicit = QCheckBox("Elicit")
        self.src_scopus = QCheckBox("Scopus")
        self.src_wos = QCheckBox("Web of Science")
        for cb in [self.src_pubmed, self.src_s2, self.src_openalex,
                   self.src_elicit, self.src_scopus, self.src_wos]:
            src_row.addWidget(cb)
        src_row.addStretch()
        layout.addLayout(src_row)

        limit_row = QHBoxLayout()
        lbl_paper_limit = QLabel("Paper Limit:")
        limit_row.addWidget(lbl_paper_limit)
        self.paper_limit = QSpinBox()
        self.paper_limit.setRange(1, 10000)
        self.paper_limit.setSingleStep(50)
        self.paper_limit.setValue(500)
        self.paper_limit.setSuffix(" papers")
        self.paper_limit.setToolTip(
            "Maximum papers to pull across selected search engines. "
            "The total is divided across engines, then across each engine's query lines."
        )
        lbl_paper_limit.setBuddy(self.paper_limit)
        limit_row.addWidget(self.paper_limit)
        limit_row.addStretch()
        layout.addLayout(limit_row)

        # Year range + options
        yr_row = QHBoxLayout()
        lbl_min = QLabel("Min Year:")
        yr_row.addWidget(lbl_min)
        self.min_year = QSpinBox()
        self.min_year.setRange(1900, 2030)
        self.min_year.setValue(2000)
        lbl_min.setBuddy(self.min_year)
        yr_row.addWidget(self.min_year)
        lbl_max = QLabel("Max Year:")
        yr_row.addWidget(lbl_max)
        self.max_year = QSpinBox()
        self.max_year.setRange(1900, 2030)
        self.max_year.setValue(2026)
        lbl_max.setBuddy(self.max_year)
        yr_row.addWidget(self.max_year)

        self.fulltext_only = QCheckBox("Full text only")
        self.fulltext_only.setToolTip(
            "Filter to papers with freely available full text (open access). "
            "Uses PubMed free full text filter and Unpaywall OA lookup."
        )
        yr_row.addWidget(self.fulltext_only)

        self.fetch_fulltext = QCheckBox("Download full text")
        self.fetch_fulltext.setToolTip(
            "Search + download in one step: when checked, Run Sysys Search "
            "automatically downloads full text for the corpus after the search "
            "completes (same as clicking Fetch Articles afterwards)."
        )
        yr_row.addWidget(self.fetch_fulltext)

        yr_row.addStretch()
        layout.addLayout(yr_row)

        # Search / Fetch / Analysis buttons
        fetch_row = QHBoxLayout()
        self.sysys_search_btn = QPushButton("Run Sysys Search")
        self.sysys_search_btn.clicked.connect(self._start_sysys_search)
        self.fetch_btn = QPushButton("Fetch Articles")
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setToolTip(
            "Download full text for the papers already in the corpus. "
            "This does not run a new search — use Run Sysys Search first to "
            "build the corpus."
        )
        self.fetch_btn.clicked.connect(self._start_fetch)
        self.analysis_btn = QPushButton("Run Keyword Analysis")
        self.analysis_btn.setEnabled(False)
        self.analysis_btn.setToolTip(
            "Available after fetched search results exist. It updates the query-term weights."
        )
        self.analysis_btn.clicked.connect(self._start_analysis)
        self.sysys_cancel_btn = QPushButton("Cancel Search")
        self.sysys_cancel_btn.setEnabled(False)
        self.sysys_cancel_btn.clicked.connect(self._cancel_sysys_search)
        self.cancel_fetch_btn = QPushButton("Cancel Fetch")
        self.cancel_fetch_btn.setEnabled(False)
        self.cancel_fetch_btn.clicked.connect(self._cancel_work)
        fetch_row.addWidget(self.sysys_search_btn)
        fetch_row.addWidget(self.fetch_btn)
        fetch_row.addWidget(self.analysis_btn)
        fetch_row.addWidget(self.sysys_cancel_btn)
        fetch_row.addWidget(self.cancel_fetch_btn)
        fetch_row.addStretch()
        layout.addLayout(fetch_row)

        self.search_progress = QProgressBar()
        self.search_progress.setVisible(False)
        layout.addWidget(self.search_progress)

        self.search_status = QLabel("Ready")
        layout.addWidget(self.search_status)

        # Feature E: inline search results preview
        self.search_results_display = QPlainTextEdit()
        self.search_results_display.setReadOnly(True)
        self.search_results_display.setAccessibleName("Search results preview")
        self.search_results_display.setPlaceholderText(
            "Search results will appear here after search completes…"
        )
        self.search_results_display.setMaximumHeight(200)
        layout.addWidget(self.search_results_display)

        layout.addStretch()
        self._add_help_button(tab, "search")
        return tab

    # ── Tab 1: Score & Filter ─────────────────────────────────────────

    def _build_score_filter_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        json_f = "JSON files (*.json);;All files (*)"

        # ── Top half: two columns ──────────────────────────────────────
        #   left  = BRT-BERT Score & Filter (Stage 2)
        #   right = Stage 2c Statistical Evidence Extraction
        top_row = QHBoxLayout()

        # ---- Left column: BRT-BERT Score & Filter ----
        left_col = QVBoxLayout()
        left_col.addWidget(
            SectionHeader("BRT-BERT Score & Filter", "Stage 2: relevance screening")
        )

        # Workload selector
        wl_row = QHBoxLayout()
        lbl_workload = QLabel("Workload:")
        wl_row.addWidget(lbl_workload)
        self.score_workload = QComboBox()
        self.score_workload.addItems(["abstract", "fulltext"])
        lbl_workload.setBuddy(self.score_workload)
        wl_row.addWidget(self.score_workload)
        wl_row.addStretch()
        left_col.addLayout(wl_row)

        # Simple-mode info label (shown by default; BRT-BERT is the default)
        self.score_simple_label = QLabel(
            "Default relevance model: BRT-BERT (BERT Relevance Trained). "
            "(change via File → Settings → “Show advanced LLM options”)"
        )
        self.score_simple_label.setStyleSheet("color: #444; font-style: italic;")
        self.score_simple_label.setWordWrap(True)
        left_col.addWidget(self.score_simple_label)

        # Advanced container — provider/model row + relevance-model dropdown.
        # Hidden by default; revealed when the user opts into advanced LLM
        # options in Settings (QSettings key score/show_advanced_llm).
        self.score_advanced_box = QWidget()
        adv_layout = QVBoxLayout(self.score_advanced_box)
        adv_layout.setContentsMargins(0, 0, 0, 0)

        row, self.score_provider, self.score_model = _make_llm_backend_row(
            endpoint_getter=self._custom_endpoint
        )
        self.score_provider.addItem("brt-bert")
        adv_layout.addLayout(row)

        rm_row = QHBoxLayout()
        rm_lbl = QLabel("Relevance model:")
        rm_row.addWidget(rm_lbl)
        self.score_relevance_model = QComboBox()
        # brt-bert (resolves to brt-bert-relevance/final) is the DEFAULT
        # relevance scorer for Stage 2; other models remain selectable.
        self.score_relevance_model.setEditable(True)
        self.score_relevance_model.addItems(["brt-bert", "housecatbert-scibert"])
        self.score_relevance_model.setCurrentText(
            getattr(self.pipeline_config, "relevance_model", "") or "brt-bert"
        )
        # Persist user edits against programmatic syncs (see _sync_settings_to_tabs).
        setattr(self.score_relevance_model, "_user_customized", False)
        self.score_relevance_model.activated.connect(
            lambda *_: setattr(self.score_relevance_model, "_user_customized", True)
        )
        relevance_model_line_edit = self.score_relevance_model.lineEdit()
        if relevance_model_line_edit is not None:
            relevance_model_line_edit.textEdited.connect(
                lambda *_: setattr(self.score_relevance_model, "_user_customized", True)
            )
        rm_lbl.setBuddy(self.score_relevance_model)
        rm_row.addWidget(self.score_relevance_model, 1)
        adv_layout.addLayout(rm_row)

        left_col.addWidget(self.score_advanced_box)

        # Threshold / batch
        param_row = QHBoxLayout()
        lbl_threshold = QLabel("Threshold:")

        param_row.addWidget(lbl_threshold)
        self.score_threshold = QSpinBox()
        self.score_threshold.setRange(1, 10)
        self.score_threshold.setValue(3)
        lbl_threshold.setBuddy(self.score_threshold)
        param_row.addWidget(self.score_threshold)
        lbl_batch = QLabel("Batch Size:")
        param_row.addWidget(lbl_batch)
        self.score_batch = QSpinBox()
        self.score_batch.setRange(1, 50)
        self.score_batch.setValue(1)
        lbl_batch.setBuddy(self.score_batch)
        param_row.addWidget(self.score_batch)
        param_row.addStretch()
        left_col.addLayout(param_row)

        # Corpus path
        cp_row, self.score_corpus_path, cp_btn = _make_file_row(
            "Corpus Path:", "path/to/corpus.json"
        )
        left_col.addLayout(cp_row)
        cp_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.score_corpus_path, json_f, self.pipeline_config.corpus_dir
            )
        )

        # Run / Cancel
        btn_row = QHBoxLayout()
        self.score_run_btn = QPushButton("Run Score & Filter")
        self.score_run_btn.clicked.connect(self._start_sysys_score)
        self.score_cancel_btn = QPushButton("Cancel")
        self.score_cancel_btn.setEnabled(False)
        self.score_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        btn_row.addWidget(self.score_run_btn)
        btn_row.addWidget(self.score_cancel_btn)
        btn_row.addStretch()
        left_col.addLayout(btn_row)

        self.score_progress = QProgressBar()
        self.score_progress.setVisible(False)
        left_col.addWidget(self.score_progress)
        self.score_status = QLabel("")
        left_col.addWidget(self.score_status)
        left_col.addStretch()

        # ---- Right column: Stage 2c Statistical Evidence Extraction ----
        right_col = QVBoxLayout()
        right_col.addWidget(
            SectionHeader("Stage 2c: Statistical Evidence Extraction")
        )
        stat_in_row, self.stat_extract_input, stat_in_btn = _make_file_row(
            "Scored Corpus:", "path/to/relevant.json"
        )
        right_col.addLayout(stat_in_row)
        stat_out_row, self.stat_extract_output, stat_out_btn = _make_file_row(
            "Extraction Output:", "path/to/statistical_extractions.json", browse_save=True
        )
        right_col.addLayout(stat_out_row)
        stat_in_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.stat_extract_input, json_f, self.pipeline_config.claims_dir
            )
        )
        stat_out_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.stat_extract_output, json_f, self.pipeline_config.claims_dir
            )
        )
        stat_llm_row, self.stat_extract_provider, self.stat_extract_model = (
            _make_llm_backend_row(
                endpoint_getter=self._custom_endpoint
            )
        )
        right_col.addLayout(stat_llm_row)

        stat_options = QHBoxLayout()
        self.stat_extract_llm_context = QCheckBox("LLM context check")
        self.stat_extract_llm_context.toggled.connect(
            self._on_stat_extract_llm_context_toggled
        )
        stat_options.addWidget(self.stat_extract_llm_context)
        lbl_stat_max = QLabel("Max contexts:")
        stat_options.addWidget(lbl_stat_max)
        self.stat_extract_max_contexts = QSpinBox()
        self.stat_extract_max_contexts.setRange(1, 999)
        self.stat_extract_max_contexts.setValue(80)
        lbl_stat_max.setBuddy(self.stat_extract_max_contexts)
        stat_options.addWidget(self.stat_extract_max_contexts)
        self.stat_extract_full_llm = QCheckBox("LLM full statistical extraction")
        self.stat_extract_full_llm.setToolTip(
            "Review every deterministic Stage 2c statistical candidate with the selected LLM."
        )
        self.stat_extract_full_llm.toggled.connect(
            self._on_stat_extract_full_llm_toggled
        )
        stat_options.addWidget(self.stat_extract_full_llm)
        stat_options.addStretch()
        right_col.addLayout(stat_options)

        stat_btn_row = QHBoxLayout()
        self.stat_extract_btn = QPushButton("Run Statistical Extraction")
        self.stat_extract_btn.clicked.connect(self._start_statistical_extract)
        self.stat_extract_cancel_btn = QPushButton("Cancel")
        self.stat_extract_cancel_btn.setEnabled(False)
        self.stat_extract_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        stat_btn_row.addWidget(self.stat_extract_btn)
        stat_btn_row.addWidget(self.stat_extract_cancel_btn)
        stat_btn_row.addStretch()
        right_col.addLayout(stat_btn_row)

        self.stat_extract_progress = QProgressBar()
        self.stat_extract_progress.setVisible(False)
        right_col.addWidget(self.stat_extract_progress)
        self.stat_extract_status = QLabel("")
        right_col.addWidget(self.stat_extract_status)
        right_col.addStretch()

        top_row.addLayout(left_col, 1)
        top_row.addLayout(right_col, 1)
        outer.addLayout(top_row, 1)

        # ── Bottom half: Human Relevance Calibration (full width) ───────
        human_group = QGroupBox("Human Relevance Calibration")
        human_layout = QVBoxLayout(human_group)

        human_controls = QHBoxLayout()
        lbl_human_sample = QLabel("Sample Size:")
        human_controls.addWidget(lbl_human_sample)
        self.human_review_sample = QSpinBox()
        self.human_review_sample.setRange(0, 500)
        self.human_review_sample.setSpecialValueText("Auto")
        self.human_review_sample.setValue(0)
        lbl_human_sample.setBuddy(self.human_review_sample)
        human_controls.addWidget(self.human_review_sample)
        self.human_review_build_btn = QPushButton("Build Queue")
        self.human_review_build_btn.clicked.connect(self._build_human_relevance_queue)
        self.human_review_edit_cells = QCheckBox("Edit Cells")
        self.human_review_edit_cells.setToolTip(
            "Allow corrections to the title, year, model score, ID, and abstract cells"
        )
        self.human_review_edit_cells.toggled.connect(
            self._set_human_relevance_editing
        )
        self.human_review_save_btn = QPushButton("Save Review")
        self.human_review_save_btn.clicked.connect(self._save_human_relevance_review)
        self.human_review_compare_btn = QPushButton("Compare to Model Scores")
        self.human_review_compare_btn.clicked.connect(self._compare_human_relevance_review)
        human_controls.addWidget(self.human_review_build_btn)
        human_controls.addWidget(self.human_review_edit_cells)
        human_controls.addWidget(self.human_review_save_btn)
        human_controls.addWidget(self.human_review_compare_btn)
        human_controls.addStretch()
        human_layout.addLayout(human_controls)

        self.human_review_table = QTableWidget(0, 7)
        self.human_review_table.setHorizontalHeaderLabels(
            [
                "Human Score",
                "Human Verified",
                "Title",
                "Year",
                "Model Score",
                "ID",
                "Abstract",
            ]
        )
        self.human_review_table.setAccessibleName("Human relevance review table")
        self.human_review_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.human_review_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.human_review_table.verticalHeader().setVisible(False)
        human_header = self.human_review_table.horizontalHeader()
        human_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        human_header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.human_review_table.setMinimumHeight(220)
        human_layout.addWidget(self.human_review_table)

        self.human_review_status = QLabel("No human queue loaded")
        human_layout.addWidget(self.human_review_status)
        outer.addWidget(human_group, 1)

        self._human_review_artifact: dict | None = None
        self._human_review_rows: list[dict] = []
        self._loaded_human_review_path: Path | None = None

        self._add_help_button(tab, "score_filter")
        return tab

    # ── Tab 2: Extract & Validate ─────────────────────────────────────

    def _build_extract_validate_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        json_f = "JSON files (*.json);;All files (*)"

        # Stage 3: Extract
        layout.addWidget(SectionHeader("Stage 3: Claim Extraction"))
        e_row, self.extract_corpus, e_btn = _make_file_row(
            "Scored Corpus:", "path/to/relevant.json"
        )
        layout.addLayout(e_row)
        eo_row, self.extract_output, eo_btn = _make_file_row(
            "Claims Output:", "path/to/claims.json", browse_save=True
        )
        layout.addLayout(eo_row)
        e_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.extract_corpus, json_f, self.pipeline_config.claims_dir
            )
        )
        eo_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.extract_output, json_f, self.pipeline_config.claims_dir
            )
        )

        er, self.extract_provider, self.extract_model = _make_llm_backend_row(
            endpoint_getter=self._custom_endpoint
        )
        layout.addLayout(er)

        ex_params = QHBoxLayout()
        lbl_max_tokens = QLabel("Max Tokens:")
        ex_params.addWidget(lbl_max_tokens)
        self.extract_max_tokens = QSpinBox()
        self.extract_max_tokens.setRange(256, 65536)
        self.extract_max_tokens.setValue(4096)
        lbl_max_tokens.setBuddy(self.extract_max_tokens)
        ex_params.addWidget(self.extract_max_tokens)
        self.extract_thinking = QCheckBox("Use Thinking")
        ex_params.addWidget(self.extract_thinking)
        ex_params.addStretch()
        layout.addLayout(ex_params)

        ex_btn_row = QHBoxLayout()
        self.extract_btn = QPushButton("Run Claim Extraction")
        self.extract_btn.clicked.connect(self._start_extract)
        self.extract_cancel_btn = QPushButton("Cancel")
        self.extract_cancel_btn.setEnabled(False)
        self.extract_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        ex_btn_row.addWidget(self.extract_btn)
        ex_btn_row.addWidget(self.extract_cancel_btn)
        ex_btn_row.addStretch()
        layout.addLayout(ex_btn_row)

        # Stage 4: Validate
        layout.addWidget(SectionHeader("Stage 4: Method Review / Validation"))
        v_row, self.validate_claims, v_btn = _make_file_row(
            "Claims JSON:", "path/to/claims.json"
        )
        layout.addLayout(v_row)
        vo_row, self.validate_output, vo_btn = _make_file_row(
            "Validated Output:", "path/to/validated.json", browse_save=True
        )
        layout.addLayout(vo_row)
        json_f = "JSON files (*.json);;All files (*)"
        v_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.validate_claims, json_f, self.pipeline_config.claims_dir
            )
        )
        mr_dir = self.pipeline_config.data_dir / "method_review"
        vo_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.validate_output, json_f, mr_dir, self.pipeline_config.claims_dir
            )
        )
        vr, self.validate_provider, self.validate_model = _make_llm_backend_row(
            endpoint_getter=self._custom_endpoint
        )
        layout.addLayout(vr)

        v_btn_row = QHBoxLayout()
        self.validate_btn = QPushButton("Run Method Review")
        self.validate_btn.clicked.connect(self._start_validate)
        self.validate_cancel_btn = QPushButton("Cancel")
        self.validate_cancel_btn.setEnabled(False)
        self.validate_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        v_btn_row.addWidget(self.validate_btn)
        v_btn_row.addWidget(self.validate_cancel_btn)
        v_btn_row.addStretch()
        layout.addLayout(v_btn_row)

        self.extract_validate_progress = QProgressBar()
        self.extract_validate_progress.setVisible(False)
        layout.addWidget(self.extract_validate_progress)
        self.extract_validate_status = QLabel("")
        layout.addWidget(self.extract_validate_status)

        layout.addStretch()
        self._add_help_button(tab, "extract_validate")
        return tab

    # ── Tab 3: Audit ──────────────────────────────────────────────────

    def _build_audit_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Stage 5: Audit
        layout.addWidget(SectionHeader("Stage 5: Span Audit"))
        a_row, self.audit_input, a_btn = _make_file_row(
            "Validated JSON:", "path/to/validated.json"
        )
        layout.addLayout(a_row)
        ao_row, self.audit_output, ao_btn = _make_file_row(
            "Audited Output:", "path/to/audited.json", browse_save=True
        )
        layout.addLayout(ao_row)
        json_f = "JSON files (*.json);;All files (*)"
        a_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.audit_input, json_f, self.pipeline_config.claims_dir
            )
        )
        ao_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.audit_output, json_f, self.pipeline_config.claims_dir
            )
        )

        sim_row = QHBoxLayout()
        lbl_sim_thresh = QLabel("Similarity Threshold:")
        sim_row.addWidget(lbl_sim_thresh)
        self.audit_threshold = QSlider(Qt.Orientation.Horizontal)
        self.audit_threshold.setRange(50, 100)
        lbl_sim_thresh.setBuddy(self.audit_threshold)
        self.audit_threshold.setValue(85)
        self.audit_threshold_label = QLabel("0.85")
        self.audit_threshold.valueChanged.connect(
            lambda v: self.audit_threshold_label.setText(f"{v / 100:.2f}")
        )
        sim_row.addWidget(self.audit_threshold)
        sim_row.addWidget(self.audit_threshold_label)
        layout.addLayout(sim_row)

        self.audit_btn = QPushButton("Run Span Audit")
        self.audit_btn.clicked.connect(self._start_audit)
        layout.addWidget(self.audit_btn)

        self.audit_progress = QProgressBar()
        self.audit_progress.setVisible(False)
        layout.addWidget(self.audit_progress)
        self.audit_status = QLabel("")
        layout.addWidget(self.audit_status)

        layout.addStretch()
        self._add_help_button(tab, "audit")
        return tab

    # ── Tab 4: Synthesis ──────────────────────────────────────────────

    def _build_synthesis_tab(self) -> QWidget:
        """Stage 6 gets its own tab so the cluster table, abstract,
        hypothesis rationale, source citations, and evidence network each
        have enough room to be readable."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Stage 6: Synthesize controls
        layout.addWidget(SectionHeader("Stage 6: Synthesis"))
        s_row, self.synth_input, s_btn = _make_file_row(
            "Audited JSON:", "path/to/audited.json"
        )
        layout.addLayout(s_row)
        so_row, self.synth_output, so_btn = _make_file_row(
            "Synthesis Output:", "path/to/synthesis.json", browse_save=True
        )
        layout.addLayout(so_row)
        json_f = "JSON files (*.json);;All files (*)"
        s_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.synth_input, json_f, self.pipeline_config.claims_dir
            )
        )
        so_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.synth_output, json_f, self.pipeline_config.clusters_dir
            )
        )

        sr, self.synth_provider, self.synth_model = _make_llm_backend_row(
            endpoint_getter=self._custom_endpoint
        )
        layout.addLayout(sr)

        synth_params = QHBoxLayout()
        self.synth_auto_clusters = QCheckBox("Auto clusters (silhouette score)")
        self.synth_auto_clusters.setChecked(True)
        self.synth_auto_clusters.setToolTip(
            "Sweep cluster counts and pick the one with the best silhouette "
            "score. Uncheck to set a fixed cluster count."
        )
        synth_params.addWidget(self.synth_auto_clusters)
        lbl_min_cluster = QLabel("Clusters:")
        synth_params.addWidget(lbl_min_cluster)
        self.synth_min_cluster = QSpinBox()
        self.synth_min_cluster.setRange(2, 50)
        self.synth_min_cluster.setValue(5)
        self.synth_min_cluster.setEnabled(False)
        lbl_min_cluster.setBuddy(self.synth_min_cluster)
        synth_params.addWidget(self.synth_min_cluster)
        self.synth_auto_clusters.toggled.connect(
            lambda checked: self.synth_min_cluster.setEnabled(not checked))
        lbl_max_cluster = QLabel("Max Cluster Tokens:")
        synth_params.addWidget(lbl_max_cluster)
        self.synth_max_tokens = QSpinBox()
        self.synth_max_tokens.setRange(256, 65536)
        self.synth_max_tokens.setValue(4096)
        lbl_max_cluster.setBuddy(self.synth_max_tokens)
        synth_params.addWidget(self.synth_max_tokens)
        synth_params.addStretch()
        layout.addLayout(synth_params)

        self.synth_btn = QPushButton("Run Synthesis")
        self.synth_btn.clicked.connect(self._start_synthesize)
        layout.addWidget(self.synth_btn)

        self.synth_progress = QProgressBar()
        self.synth_progress.setVisible(False)
        layout.addWidget(self.synth_progress)
        self.synth_status = QLabel("")
        layout.addWidget(self.synth_status)

        # ── Output area: cluster table on top, one roomy sub-tab per output ──
        splitter = QSplitter(Qt.Orientation.Vertical)

        cluster_group = QGroupBox("Cluster Summary")
        cluster_layout = QVBoxLayout(cluster_group)
        self.cluster_table = QTableWidget(0, 3)
        self.cluster_table.setHorizontalHeaderLabels(["Cluster", "Papers", "Key Terms"])
        self.cluster_table.horizontalHeader().setStretchLastSection(True)
        self.cluster_table.setMinimumHeight(140)
        self.cluster_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        cluster_layout.addWidget(self.cluster_table)
        splitter.addWidget(cluster_group)

        self.synth_output_tabs = QTabWidget()

        # Sub-tab: Abstract
        abstract_page = QWidget()
        abstract_layout = QVBoxLayout(abstract_page)
        self.synth_narrative = QTextEdit()
        self.synth_narrative.setAccessibleName("Synthesis narrative")
        self.synth_narrative.setPlaceholderText(
            "Run Stage 6 to generate the synthesized abstract."
        )
        abstract_layout.addWidget(self.synth_narrative, 1)
        abstract_btn_row = QHBoxLayout()
        synth_copy_btn = QPushButton("Copy Abstract")
        synth_copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(self.synth_narrative.toPlainText()))
        abstract_btn_row.addWidget(synth_copy_btn)
        abstract_btn_row.addStretch()
        abstract_layout.addLayout(abstract_btn_row)
        self.synth_output_tabs.addTab(abstract_page, "Abstract")

        # Sub-tab: Why This Hypothesis
        why_page = QWidget()
        why_layout = QVBoxLayout(why_page)
        self.hypothesis_explanation = QTextEdit()
        self.hypothesis_explanation.setReadOnly(True)
        self.hypothesis_explanation.setAccessibleName("Hypothesis explanation")
        self.hypothesis_explanation.setPlaceholderText(
            "Run Stage 6 or load hypothesis_explanation.md to inspect deterministic provenance."
        )
        why_layout.addWidget(self.hypothesis_explanation, 1)
        why_btn_row = QHBoxLayout()
        self.load_hypothesis_explanation_btn = QPushButton("Load Why This Hypothesis")
        self.load_hypothesis_explanation_btn.clicked.connect(
            self._load_hypothesis_explanation
        )
        why_btn_row.addWidget(self.load_hypothesis_explanation_btn)
        why_btn_row.addStretch()
        why_layout.addLayout(why_btn_row)
        self.synth_output_tabs.addTab(why_page, "Why This Hypothesis")

        # Sub-tab: Source Citations (its own view — no longer overwrites the abstract)
        sources_page = QWidget()
        sources_layout = QVBoxLayout(sources_page)
        self.synth_sources_view = QTextEdit()
        self.synth_sources_view.setReadOnly(True)
        self.synth_sources_view.setAccessibleName("Synthesis source citations")
        self.synth_sources_view.setPlaceholderText(
            "Run Stage 6 or load synthesis_sources.md to inspect per-claim citations."
        )
        sources_layout.addWidget(self.synth_sources_view, 1)
        sources_btn_row = QHBoxLayout()
        self.load_synthesis_sources_btn = QPushButton("Load Source Citations")
        self.load_synthesis_sources_btn.clicked.connect(self._load_synthesis_sources)
        sources_btn_row.addWidget(self.load_synthesis_sources_btn)
        sources_btn_row.addStretch()
        sources_layout.addLayout(sources_btn_row)
        self.synth_output_tabs.addTab(sources_page, "Source Citations")

        # Sub-tab: Evidence Network (embedded graph view)
        network_page = QWidget()
        network_layout = QVBoxLayout(network_page)
        from gui.network_view import NetworkGraphWidget
        self.synth_network_widget = NetworkGraphWidget()
        self.synth_network_widget.setMinimumHeight(320)
        network_layout.addWidget(self.synth_network_widget, 1)
        network_btn_row = QHBoxLayout()
        self.load_evidence_network_btn = QPushButton("Load Evidence Network")
        self.load_evidence_network_btn.clicked.connect(self._load_evidence_network)
        network_btn_row.addWidget(self.load_evidence_network_btn)
        network_btn_row.addStretch()
        network_layout.addLayout(network_btn_row)
        self.synth_output_tabs.addTab(network_page, "Evidence Network")

        splitter.addWidget(self.synth_output_tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

        self._add_help_button(tab, "synthesis")
        return tab


    # ── Tab 5: Fulltext & Interrater ──────────────────────────────────

    def _build_fulltext_interrater_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Corpus full-text retrieval
        layout.addWidget(SectionHeader("Full-Text Retrieval"))
        ft_row, self.ft_output_path, ft_btn = _make_file_row(
            "JSON Output:", "path/to/fulltext.json", browse_save=True
        )
        layout.addLayout(ft_row)
        ft_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.ft_output_path,
                "JSON files (*.json);;All files (*)",
                self.pipeline_config.data_dir / "fulltext",
            )
        )

        ft_kpi = QHBoxLayout()
        self.ft_total_card = MetricCard("Total")
        self.ft_retrieved_card = MetricCard("Retrieved")
        self.ft_abstract_card = MetricCard("Abstract Only")
        self.ft_rate_card = MetricCard("Rate")
        ft_kpi.addWidget(self.ft_total_card)
        ft_kpi.addWidget(self.ft_retrieved_card)
        ft_kpi.addWidget(self.ft_abstract_card)
        ft_kpi.addWidget(self.ft_rate_card)
        layout.addLayout(ft_kpi)

        self.ft_results_table = QTableWidget(0, 4)
        self.ft_results_table.setHorizontalHeaderLabels(
            ["Article ID", "Title", "Status", "Source"]
        )
        layout.addWidget(self.ft_results_table)

        ft_btns = QHBoxLayout()
        self.ft_run_btn = QPushButton("Retrieve Full Texts")
        self.ft_run_btn.clicked.connect(self._start_fulltext_retrieval)
        self.ft_cancel_btn = QPushButton("Cancel")
        self.ft_cancel_btn.setEnabled(False)
        self.ft_cancel_btn.clicked.connect(self._cancel_ft)
        ft_btns.addWidget(self.ft_run_btn)
        ft_btns.addWidget(self.ft_cancel_btn)
        ft_btns.addStretch()
        layout.addLayout(ft_btns)

        # Stage 7: pipeline fulltext
        layout.addWidget(SectionHeader("Stage 7: Full-Text Enrichment"))
        s7_row, self.sysys_ft_corpus, s7_btn = _make_file_row(
            "Scored Corpus:", "path/to/relevant.json"
        )
        layout.addLayout(s7_row)
        json_f = "JSON files (*.json);;All files (*)"
        s7_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.sysys_ft_corpus, json_f, self.pipeline_config.claims_dir
            )
        )

        s7_w = QHBoxLayout()
        lbl_workers = QLabel("Workers:")
        s7_w.addWidget(lbl_workers)
        self.sysys_ft_workers = QSpinBox()
        self.sysys_ft_workers.setRange(1, 32)
        self.sysys_ft_workers.setValue(3)
        lbl_workers.setBuddy(self.sysys_ft_workers)
        s7_w.addWidget(self.sysys_ft_workers)
        s7_w.addStretch()
        layout.addLayout(s7_w)

        ft_options = QGroupBox("Memory, Resume && Indexing")
        ft_options_form = QFormLayout(ft_options)

        self.sysys_ft_streaming = QCheckBox(
            "Stream one paper at a time and resume from JSONL"
        )
        self.sysys_ft_streaming.setChecked(True)
        self.sysys_ft_streaming.setToolTip(
            "Preserves completed Stage 7 records across cancellation or restart."
        )
        ft_options_form.addRow("Stage 7:", self.sysys_ft_streaming)

        self.sysys_ft_batch_size = QSpinBox()
        self.sysys_ft_batch_size.setRange(1, 10000)
        self.sysys_ft_batch_size.setValue(50)
        self.sysys_ft_batch_size.setToolTip(
            "Maximum papers held by the legacy buffered path. Streaming mode "
            "always processes one paper at a time."
        )
        ft_options_form.addRow("Buffered Batch:", self.sysys_ft_batch_size)

        self.sysys_ft_paperqa_enabled = QCheckBox(
            "Build PaperQA index during corpus full-text enrichment"
        )
        self.sysys_ft_paperqa_enabled.setChecked(True)
        self.sysys_ft_paperqa_enabled.setToolTip(
            "Applies to Download Full Text / Fetch Articles corpus enrichment. "
            "Disable to avoid loading the optional PaperQA document store."
        )
        ft_options_form.addRow("PaperQA:", self.sysys_ft_paperqa_enabled)

        self.sysys_ft_paperqa_batch_size = QSpinBox()
        self.sysys_ft_paperqa_batch_size.setRange(1, 10000)
        self.sysys_ft_paperqa_batch_size.setValue(100)
        self.sysys_ft_paperqa_batch_size.setToolTip(
            "Checkpoint the PaperQA document store after this many input rows."
        )
        ft_options_form.addRow(
            "Index Checkpoint Batch:", self.sysys_ft_paperqa_batch_size
        )

        for control in (
            self.sysys_ft_workers,
            self.sysys_ft_streaming,
            self.sysys_ft_batch_size,
            self.sysys_ft_paperqa_enabled,
            self.sysys_ft_paperqa_batch_size,
        ):
            signal = (
                control.toggled
                if isinstance(control, QCheckBox)
                else control.valueChanged
            )
            signal.connect(self._sync_fulltext_controls_to_config)
        self.sysys_ft_streaming.toggled.connect(
            lambda enabled: self.sysys_ft_batch_size.setDisabled(enabled)
        )
        self.sysys_ft_paperqa_enabled.toggled.connect(
            self.sysys_ft_paperqa_batch_size.setEnabled
        )
        self.sysys_ft_batch_size.setDisabled(True)
        layout.addWidget(ft_options)

        self.sysys_ft_resume = QCheckBox("Resume completed papers")
        self.sysys_ft_resume.setChecked(True)
        self.sysys_ft_resume.setToolTip(
            "Reuse successful download and metric checkpoints. "
            "Uncheck to retrieve all selected papers again."
        )
        layout.addWidget(self.sysys_ft_resume)
        self.sysys_ft_btn = QPushButton("Run Full-Text Enrichment")
        self.sysys_ft_btn.clicked.connect(self._start_sysys_fulltext)
        layout.addWidget(self.sysys_ft_btn)
        self.sysys_ft_cancel_btn = QPushButton("Cancel Full-Text Enrichment")
        self.sysys_ft_cancel_btn.setEnabled(False)
        self.sysys_ft_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        layout.addWidget(self.sysys_ft_cancel_btn)

        # Stage 8: Interrater
        layout.addWidget(SectionHeader("Stage 8: Interrater Scoring"))
        i_row, self.interrater_corpus, i_btn = _make_file_row(
            "Scored Corpus:", "path/to/relevant.json"
        )
        layout.addLayout(i_row)
        json_f = "JSON files (*.json);;All files (*)"
        i_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.interrater_corpus, json_f, self.pipeline_config.claims_dir
            )
        )
        ir, self.interrater_provider, self.interrater_model = _make_llm_backend_row(
            default_provider="brt-irr",
            extra_providers=["brt-irr"],
            endpoint_getter=self._custom_endpoint,
        )
        layout.addLayout(ir)
        brt_hint = QLabel(
            "brt-irr = brt-bert-irr-buddy-mlm (model fine-tuned for IRR, score-only). "
            "Default and only BERT option for Stage 8. "
            "Leave Model blank to use the preset path, or enter a custom model path/HF id."
        )
        brt_hint.setWordWrap(True)
        layout.addWidget(brt_hint)

        i_params = QHBoxLayout()
        lbl_sample = QLabel("Sample Size:")
        i_params.addWidget(lbl_sample)
        self.interrater_sample = QSpinBox()
        self.interrater_sample.setRange(1, 500)
        self.interrater_sample.setValue(20)
        lbl_sample.setBuddy(self.interrater_sample)
        i_params.addWidget(self.interrater_sample)
        i_params.addStretch()
        layout.addLayout(i_params)

        self.interrater_btn = QPushButton("Run Interrater Scoring")
        self.interrater_btn.clicked.connect(self._start_interrater)
        layout.addWidget(self.interrater_btn)
        self.interrater_cancel_btn = QPushButton("Cancel")
        self.interrater_cancel_btn.setEnabled(False)
        self.interrater_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        layout.addWidget(self.interrater_cancel_btn)

        self.ft_ir_progress = QProgressBar()
        self.ft_ir_progress.setVisible(False)
        layout.addWidget(self.ft_ir_progress)
        self.ft_ir_status = QLabel("")
        layout.addWidget(self.ft_ir_status)

        layout.addStretch()
        self._add_help_button(tab, "fulltext_interrater")
        return tab

    # ── Tab 6: Robustness & Verify ────────────────────────────────────

    def _build_robustness_verify_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Stage 9: Robustness
        layout.addWidget(SectionHeader("Stage 9: Robustness Checks"))
        r_row, self.robust_input, r_btn = _make_file_row(
            "Audited Corpus:", "path/to/audited.json"
        )
        layout.addLayout(r_row)
        ro_row, self.robust_output, ro_btn = _make_file_row(
            "Report Output:", "path/to/robustness.json", browse_save=True
        )
        layout.addLayout(ro_row)
        json_f = "JSON files (*.json);;All files (*)"
        val_dir = self.pipeline_config.data_dir / "validation"
        r_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.robust_input, json_f, self.pipeline_config.claims_dir
            )
        )
        ro_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.robust_output, json_f, val_dir, self.pipeline_config.claims_dir
            )
        )

        self.robust_btn = QPushButton("Run Robustness Checks")
        self.robust_btn.clicked.connect(self._start_robustness)
        layout.addWidget(self.robust_btn)

        # Stage 10: Verify
        layout.addWidget(SectionHeader("Stage 10: Abstract Verification"))
        v_row, self.verify_claims, v_btn = _make_file_row(
            "Claims/Synthesis:", "path/to/claims.json"
        )
        layout.addLayout(v_row)
        json_f = "JSON files (*.json);;All files (*)"
        v_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.verify_claims, json_f, self.pipeline_config.claims_dir
            )
        )
        vr, self.verify_provider, self.verify_model = _make_llm_backend_row(
            endpoint_getter=self._custom_endpoint
        )
        layout.addLayout(vr)

        self.verify_btn = QPushButton("Run Abstract Verification")
        self.verify_btn.clicked.connect(self._start_verify)
        layout.addWidget(self.verify_btn)

        self.robust_verify_progress = QProgressBar()
        self.robust_verify_progress.setVisible(False)
        layout.addWidget(self.robust_verify_progress)
        self.robust_verify_status = QLabel("")
        layout.addWidget(self.robust_verify_status)

        self.robust_results = QTextEdit()
        self.robust_results.setReadOnly(True)
        self.robust_results.setAccessibleName("Robustness verification results")
        layout.addWidget(self.robust_results)

        # Inline verification report viewer
        layout.addWidget(SectionHeader("Verification Report"))
        self.verify_report_view = QTextBrowser()
        self.verify_report_view.setMinimumHeight(260)
        self.verify_report_view.setAccessibleName("Verification report")
        self.verify_report_view.setPlaceholderText(
            "Run Stage 10 (Abstract Verification) to see the full report here.")
        self.verify_report_view.setOpenExternalLinks(False)
        layout.addWidget(self.verify_report_view)

        # ── Meta-Analytic Synthesis section ──
        layout.addWidget(SectionHeader("Meta-Analytic Synthesis"))

        meta_ctrl = QHBoxLayout()
        self.meta_analysis_btn = QPushButton("Run Meta-Analysis")
        self.meta_analysis_btn.clicked.connect(self._start_meta_analysis)
        meta_ctrl.addWidget(self.meta_analysis_btn)
        self.meta_analysis_status = QLabel("Not yet run.")
        meta_ctrl.addWidget(self.meta_analysis_status, 1)
        layout.addLayout(meta_ctrl)

        # Plot tabs: Forest plot / Funnel plot
        self.meta_plot_tabs = QTabWidget()
        self.meta_plot_tabs.setMinimumHeight(350)

        # Forest plot placeholder
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        forest_fig = Figure(figsize=(10, 5), dpi=100, facecolor="white")
        forest_ax = forest_fig.add_subplot(111)
        forest_ax.text(0.5, 0.5, "Click 'Run Meta-Analysis' to generate forest plot",
                       ha="center", va="center", fontsize=11, color="#808080")
        forest_ax.axis("off")
        self.forest_canvas = FigureCanvasQTAgg(forest_fig)
        self.meta_plot_tabs.addTab(self.forest_canvas, "Forest Plot")

        # Funnel plot placeholder
        funnel_fig = Figure(figsize=(7, 6), dpi=100, facecolor="white")
        funnel_ax = funnel_fig.add_subplot(111)
        funnel_ax.text(0.5, 0.5, "Click 'Run Meta-Analysis' to generate funnel plot",
                       ha="center", va="center", fontsize=11, color="#808080")
        funnel_ax.axis("off")
        self.funnel_canvas = FigureCanvasQTAgg(funnel_fig)
        self.meta_plot_tabs.addTab(self.funnel_canvas, "Funnel Plot")

        layout.addWidget(self.meta_plot_tabs)

        # Summary statistics display
        self.meta_summary = QTextEdit()
        self.meta_summary.setReadOnly(True)
        self.meta_summary.setAccessibleName("Meta-analysis summary")
        self.meta_summary.setMaximumHeight(180)
        self.meta_summary.setPlaceholderText(
            "Meta-analysis summary statistics will appear here."
        )
        layout.addWidget(self.meta_summary)

        self._add_help_button(tab, "robustness_verify")
        return tab

    # ── Tab 7: Text Scaling ───────────────────────────────────────────

    def _build_scaling_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Upper: Wordfish
        upper = QWidget()
        u_layout = QVBoxLayout(upper)
        u_layout.addWidget(SectionHeader("Wordfish Scaling"))

        wf_params = QHBoxLayout()
        lbl_min_words = QLabel("Min Doc Words:")
        wf_params.addWidget(lbl_min_words)
        self.wf_min_words = QSpinBox()
        self.wf_min_words.setRange(10, 10000)
        self.wf_min_words.setValue(50)
        lbl_min_words.setBuddy(self.wf_min_words)
        wf_params.addWidget(self.wf_min_words)
        lbl_max_features = QLabel("Max Features:")
        wf_params.addWidget(lbl_max_features)
        self.wf_max_features = QSpinBox()
        self.wf_max_features.setRange(100, 100000)
        self.wf_max_features.setValue(5000)
        lbl_max_features.setBuddy(self.wf_max_features)
        wf_params.addWidget(self.wf_max_features)
        wf_params.addStretch()
        u_layout.addLayout(wf_params)

        wf_actions = QHBoxLayout()
        self.wf_run_btn = QPushButton("Run Text Scaling")
        self.wf_run_btn.clicked.connect(self._start_text_scaling)
        wf_actions.addWidget(self.wf_run_btn)
        self.wf_cancel_btn = QPushButton("Cancel")
        self.wf_cancel_btn.setEnabled(False)
        self.wf_cancel_btn.clicked.connect(self._cancel_scaling)
        wf_actions.addWidget(self.wf_cancel_btn)
        wf_actions.addStretch()
        u_layout.addLayout(wf_actions)

        wf_kpi = QHBoxLayout()
        self.wf_docs_card = MetricCard("Documents")
        self.wf_features_card = MetricCard("Features")
        self.wf_convergence_card = MetricCard("Convergence")
        wf_kpi.addWidget(self.wf_docs_card)
        wf_kpi.addWidget(self.wf_features_card)
        wf_kpi.addWidget(self.wf_convergence_card)
        u_layout.addLayout(wf_kpi)

        self.wf_table = QTableWidget(0, 3)
        self.wf_table.setHorizontalHeaderLabels(["Article", "Position", "Direction"])
        self.wf_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.wf_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.wf_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        u_layout.addWidget(self.wf_table)

        self.wf_terms_detail = QTextEdit()
        self.wf_terms_detail.setReadOnly(True)
        self.wf_terms_detail.setAccessibleName("Wordfish discriminating terms")
        self.wf_terms_detail.setMaximumHeight(72)
        self.wf_terms_detail.setPlaceholderText(
            "The vocabulary most associated with each pole will appear here."
        )
        u_layout.addWidget(self.wf_terms_detail)

        splitter.addWidget(upper)

        # Lower: Stylometry
        lower = QWidget()
        l_layout = QVBoxLayout(lower)
        l_layout.addWidget(SectionHeader("Stylometric Profiles"))

        sty_kpi = QHBoxLayout()
        self.sty_corpus_card = MetricCard("Corpus")
        self.sty_avg_hedge_card = MetricCard("Avg Hedge")
        self.sty_avg_certainty_card = MetricCard("Avg Certainty")
        sty_kpi.addWidget(self.sty_corpus_card)
        sty_kpi.addWidget(self.sty_avg_hedge_card)
        sty_kpi.addWidget(self.sty_avg_certainty_card)
        l_layout.addLayout(sty_kpi)

        self.sty_table = QTableWidget(0, 5)
        self.sty_table.setHorizontalHeaderLabels(
            ["Article", "Hedge", "Certainty", "FK Grade", "TTR"]
        )
        self.sty_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            self.sty_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.sty_table.cellClicked.connect(self._on_sty_row_clicked)
        l_layout.addWidget(self.sty_table)

        self.sty_detail = QTextEdit()
        self.sty_detail.setReadOnly(True)
        self.sty_detail.setAccessibleName("Stylometric details")
        self.sty_detail.setMaximumHeight(120)
        l_layout.addWidget(self.sty_detail)

        splitter.addWidget(lower)
        layout.addWidget(splitter)
        self._add_help_button(tab, "text_scaling")
        return tab

    # ── Tab 8: Evidence Map ───────────────────────────────────────────

    def _build_evidence_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Upper: evidence table
        upper = QWidget()
        u_layout = QVBoxLayout(upper)
        u_layout.addWidget(SectionHeader("Evidence Table"))

        filter_row = QHBoxLayout()
        lbl_dir = QLabel("Direction:")
        filter_row.addWidget(lbl_dir)
        self.ev_direction_filter = QComboBox()
        self.ev_direction_filter.addItems(
            ["All", "supports", "contradicts", "neutral", "tangential"]
        )
        lbl_dir.setBuddy(self.ev_direction_filter)
        filter_row.addWidget(self.ev_direction_filter)
        lbl_comp = QLabel("Component:")
        filter_row.addWidget(lbl_comp)
        self.ev_component_filter = QComboBox()
        self.ev_component_filter.addItem("All")
        lbl_comp.setBuddy(self.ev_component_filter)
        filter_row.addWidget(self.ev_component_filter)
        filter_row.addStretch()
        self.ev_refresh_btn = QPushButton("Load Evidence Table")
        self.ev_refresh_btn.setMinimumHeight(30)
        self.ev_refresh_btn.clicked.connect(self._refresh_evidence_table)
        filter_row.addWidget(self.ev_refresh_btn)
        u_layout.addLayout(filter_row)

        self.ev_status_label = QLabel("No evidence loaded.")
        self.ev_status_label.setWordWrap(True)
        u_layout.addWidget(self.ev_status_label)

        self.evidence_table = QTableWidget(0, 5)
        self.evidence_table.setHorizontalHeaderLabels(
            ["Article", "Component", "Score", "Direction", "Excerpt"]
        )
        evidence_header = self.evidence_table.horizontalHeader()
        evidence_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        evidence_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        evidence_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        evidence_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        evidence_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.evidence_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.evidence_table.cellClicked.connect(self._on_evidence_cell_clicked)
        u_layout.addWidget(self.evidence_table)

        self.ev_direction_filter.currentTextChanged.connect(
            self._apply_evidence_filters
        )
        self.ev_component_filter.currentTextChanged.connect(
            self._apply_evidence_filters
        )

        self.evidence_detail = QTextEdit()
        self.evidence_detail.setReadOnly(True)
        self.evidence_detail.setAccessibleName("Full evidence details")
        self.evidence_detail.setMaximumHeight(120)
        u_layout.addWidget(self.evidence_detail)

        splitter.addWidget(upper)

        # Lower: trend chart (placeholder)
        lower = QWidget()
        l_layout = QVBoxLayout(lower)
        l_layout.addWidget(SectionHeader("Trend Analysis"))

        trend_row = QHBoxLayout()
        lbl_trend_comp = QLabel("Component:")
        trend_row.addWidget(lbl_trend_comp)
        self.trend_component = QComboBox()
        self.trend_component.addItem("All")
        lbl_trend_comp.setBuddy(self.trend_component)
        trend_row.addWidget(self.trend_component)
        lbl_trend_metric = QLabel("Metric:")
        trend_row.addWidget(lbl_trend_metric)
        self.trend_metric = QComboBox()
        self.trend_metric.addItems(["Article Count", "Avg Support Score", "Citations"])
        lbl_trend_metric.setBuddy(self.trend_metric)
        trend_row.addWidget(self.trend_metric)
        trend_row.addStretch()
        l_layout.addLayout(trend_row)

        # Chart placeholder (matplotlib FigureCanvas would go here)
        self.trend_chart_area = QWidget()
        self.trend_chart_area.setMinimumHeight(200)
        l_layout.addWidget(self.trend_chart_area)

        splitter.addWidget(lower)
        layout.addWidget(splitter)
        self._add_help_button(tab, "evidence_map")
        return tab

    # ── Tab 9: Results Dashboard ──────────────────────────────────────

    def _build_alignment_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(SectionHeader("Results Dashboard"))

        # KPI cards
        kpi_row = QHBoxLayout()
        self.dash_total_card = MetricCard("Total Articles")
        self.dash_analyzed_card = MetricCard("Analyzed")
        self.dash_supporting_card = MetricCard("Supporting")
        self.dash_contradicting_card = MetricCard("Contradicting")
        self.dash_overall_card = MetricCard("Overall Score")
        kpi_row.addWidget(self.dash_total_card)
        kpi_row.addWidget(self.dash_analyzed_card)
        kpi_row.addWidget(self.dash_supporting_card)
        kpi_row.addWidget(self.dash_contradicting_card)
        kpi_row.addWidget(self.dash_overall_card)
        layout.addLayout(kpi_row)

        # GRADE assessment button
        grade_row = QHBoxLayout()
        self.grade_run_btn = QPushButton("Run GRADE Assessment")
        self.grade_run_btn.clicked.connect(self._run_grade_assessment)
        grade_row.addWidget(self.grade_run_btn)
        self.grade_status_label = QLabel("")
        self.grade_status_label.setStyleSheet(f"font-size: 12pt; color: {COLORS['text_secondary']};")
        grade_row.addWidget(self.grade_status_label)
        grade_row.addStretch()
        layout.addLayout(grade_row)

        # Per-component alignment bars (with GRADE badge placeholders)
        layout.addWidget(SectionHeader("Per-Component Alignment"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.alignment_bars_widget = QWidget()
        self.alignment_bars_layout = QVBoxLayout(self.alignment_bars_widget)
        scroll.setWidget(self.alignment_bars_widget)
        layout.addWidget(scroll)

        # Collapsible GRADE Summary of Findings table
        self.grade_sof_group = QGroupBox("GRADE Summary of Findings")
        self.grade_sof_group.setCheckable(True)
        self.grade_sof_group.setChecked(False)
        sof_layout = QVBoxLayout(self.grade_sof_group)

        self.grade_sof_table = QTableWidget()
        self.grade_sof_table.setColumnCount(8)
        self.grade_sof_table.setHorizontalHeaderLabels([
            "Component", "N", "Risk of Bias", "Inconsistency",
            "Indirectness", "Imprecision", "Pub. Bias", "Certainty",
        ])
        self.grade_sof_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.grade_sof_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.grade_sof_table.currentCellChanged.connect(self._on_grade_row_selected)
        sof_layout.addWidget(self.grade_sof_table)
        self.grade_sof_group.setVisible(False)
        layout.addWidget(self.grade_sof_group)

        # Detail panel
        self.alignment_detail = QTextEdit()
        self.alignment_detail.setReadOnly(True)
        self.alignment_detail.setAccessibleName("Detailed alignment information")
        self.alignment_detail.setMaximumHeight(150)
        layout.addWidget(self.alignment_detail)

        # Internal state
        self._grade_badges: list[GradeBadge] = []
        self._grade_results: list = []  # list[ComponentGrade]
        self._grade_sof_data: list[dict] = []

        self._add_help_button(tab, "results_dashboard")
        return tab

    # ── GRADE assessment (Results Dashboard) ────────────────────────

    def _run_grade_assessment(self):
        """Run GRADE certainty assessment and update the dashboard."""
        from process.grade import grade_all_components, generate_sof_table

        self.grade_status_label.setText("Running GRADE assessment...")
        QApplication.processEvents()

        try:
            self._grade_results = grade_all_components(self.pipeline_config)

            if not self._grade_results:
                self.grade_status_label.setText(
                    "No components found. Run the pipeline first."
                )
                return

            # Build SoF table data
            self._grade_sof_data = generate_sof_table(self._grade_results)
            self._populate_grade_sof_table(self._grade_sof_data)
            self.grade_sof_group.setVisible(True)
            self.grade_sof_group.setChecked(True)

            self._update_alignment_dashboard()

            self.grade_status_label.setText(
                f"GRADE complete — {len(self._grade_results)} components assessed."
            )
            self._log("GRADE evidence assessment completed.")
        except Exception as exc:
            self.grade_status_label.setText(f"Error: {exc}")
            self._log(f"GRADE assessment failed: {exc}", level="ERROR")
            logger.exception("GRADE assessment failed")

    def _attach_grade_badges(self):
        """Add or update GradeBadge widgets next to each AlignmentBar row."""
        # Build a lookup from component_id -> certainty
        certainty_map: dict[str, str] = {}
        for g in self._grade_results:
            certainty_map[g.component_id] = g.overall_certainty.value

        # Walk the alignment_bars_layout and attach badges
        for i in range(self.alignment_bars_layout.count()):
            item = self.alignment_bars_layout.itemAt(i)
            if item is None:
                continue
            widget = item.widget()
            if widget is None:
                # Might be a layout
                sub_layout = item.layout()
                if sub_layout is None:
                    continue
                # Check if this row already has a badge (3rd widget)
                bar_widget = None
                for j in range(sub_layout.count()):
                    sub_item = sub_layout.itemAt(j)
                    w = sub_item.widget() if sub_item is not None else None
                    if isinstance(w, AlignmentBar):
                        bar_widget = w
                        break
                if bar_widget is not None:
                    # Try to find the component_id from the bar label
                    comp_id = getattr(bar_widget, "_component_id", None)
                    if comp_id and comp_id in certainty_map:
                        # Check for existing badge in this layout
                        existing_badge = None
                        for j in range(sub_layout.count()):
                            sub_item = sub_layout.itemAt(j)
                            w = sub_item.widget() if sub_item is not None else None
                            if isinstance(w, GradeBadge):
                                existing_badge = w
                                break
                        if existing_badge:
                            existing_badge.set_certainty(certainty_map[comp_id])
                        else:
                            badge = GradeBadge(certainty_map[comp_id])
                            sub_layout.addWidget(badge)
                            self._grade_badges.append(badge)
                continue

            # If the widget itself is an AlignmentBar in a direct layout, wrap it
            if isinstance(widget, AlignmentBar):
                comp_id = getattr(widget, "_component_id", None)
                if comp_id and comp_id in certainty_map:
                    # Replace the widget with an HBoxLayout containing bar + badge
                    self.alignment_bars_layout.removeWidget(widget)
                    row_layout = QHBoxLayout()
                    row_layout.addWidget(widget, stretch=1)
                    badge = GradeBadge(certainty_map[comp_id])
                    row_layout.addWidget(badge)
                    self._grade_badges.append(badge)
                    self.alignment_bars_layout.insertLayout(i, row_layout)

    def _populate_grade_sof_table(self, sof_data: list[dict]):
        """Fill the GRADE Summary of Findings QTableWidget."""
        domain_keys = [
            "risk_of_bias", "inconsistency", "indirectness",
            "imprecision", "publication_bias",
        ]
        concern_colors = {
            "no_concern": "#c8e6c9",   # light green
            "serious": "#fff9c4",       # light yellow
            "very_serious": "#ffcdd2",  # light red
        }

        table = self.grade_sof_table
        table.setRowCount(len(sof_data))

        for row, entry in enumerate(sof_data):
            # Component name
            comp_item = QTableWidgetItem(entry.get("component", ""))
            comp_item.setFlags(comp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, comp_item)

            # N studies
            n_item = QTableWidgetItem(str(entry.get("n_studies", 0)))
            n_item.setFlags(n_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            n_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row, 1, n_item)

            # Domain columns
            for col_idx, dk in enumerate(domain_keys, start=2):
                d = entry.get(dk, {})
                label = d.get("label", "N/A") if isinstance(d, dict) else str(d)
                concern = d.get("concern", "no_concern") if isinstance(d, dict) else "no_concern"

                cell = QTableWidgetItem(label)
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                bg = concern_colors.get(concern, "#ffffff")
                cell.setBackground(QColor(bg))
                table.setItem(row, col_idx, cell)

            # Overall certainty
            cert = entry.get("overall_certainty", "")
            cert_item = QTableWidgetItem(cert)
            cert_item.setFlags(cert_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            cert_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            cert_bg = GradeBadge.COLORS.get(cert, "#ffffff")
            cert_item.setBackground(QColor(cert_bg))
            cert_fg = GradeBadge.TEXT_COLORS.get(cert, "#000000")
            cert_item.setForeground(QColor(cert_fg))
            table.setItem(row, 7, cert_item)

    def _on_grade_row_selected(self, row: int, _col: int, _prev_row: int, _prev_col: int):
        """Show GRADE rationale in the detail panel when a SoF row is clicked."""
        if row < 0 or row >= len(self._grade_results):
            return
        grade = self._grade_results[row]
        lines = [f"<b>{grade.component_label}</b> — {grade.overall_certainty.value} certainty"]
        lines.append(f"<br>Starting certainty: {grade.starting_certainty} | Studies: {grade.n_studies}")
        lines.append("<br>")
        for d in grade.domains:
            icon = {
                "no_concern": "\u2705",
                "serious": "\u26a0\ufe0f",
                "very_serious": "\u274c",
            }.get(d.concern.value, "")
            lines.append(
                f"<b>{d.domain.replace('_', ' ').title()}</b>: "
                f"{icon} {d.concern.value.replace('_', ' ').title()} "
                f"(downgrade {d.downgrade_levels})<br>"
                f"<i>{d.rationale}</i><br>"
            )
        lines.append(f"<br><b>Summary:</b> {grade.summary}")
        self.alignment_detail.setHtml("".join(lines))

    # ── Tab 10: Neuroimaging ───────────────────────────────────────────

    def _build_neuroimaging_tab(self) -> QWidget:
        from gui.network_view import NetworkGraphWidget

        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        inner = QWidget()
        layout = QVBoxLayout(inner)

        # ── Section 1: Coordinate Extraction ──
        layout.addWidget(SectionHeader("Coordinate Extraction"))

        coord_ctrl = QHBoxLayout()
        self.neuro_extract_btn = QPushButton("Extract Coordinates")
        self.neuro_extract_btn.clicked.connect(self._start_coordinate_extraction)
        coord_ctrl.addWidget(self.neuro_extract_btn)
        self.neuro_extract_status = QLabel("No coordinates extracted yet.")
        coord_ctrl.addWidget(self.neuro_extract_status, 1)
        layout.addLayout(coord_ctrl)

        # Coordinate results table
        self.neuro_coord_table = QTableWidget(0, 8)
        self.neuro_coord_table.setHorizontalHeaderLabels([
            "Article", "X", "Y", "Z", "Space", "Region", "Stat", "N",
        ])
        self.neuro_coord_table.horizontalHeader().setStretchLastSection(True)
        self.neuro_coord_table.setAlternatingRowColors(True)
        self.neuro_coord_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.neuro_coord_table)

        # Export buttons
        export_row = QHBoxLayout()
        self.neuro_export_nimare_btn = QPushButton("Export NiMARE")
        self.neuro_export_nimare_btn.clicked.connect(self._export_nimare)
        self.neuro_export_nimare_btn.setEnabled(False)
        export_row.addWidget(self.neuro_export_nimare_btn)
        self.neuro_export_ale_btn = QPushButton("Export GingerALE")
        self.neuro_export_ale_btn.clicked.connect(self._export_ginger_ale)
        self.neuro_export_ale_btn.setEnabled(False)
        export_row.addWidget(self.neuro_export_ale_btn)
        self.neuro_export_csv_btn = QPushButton("Export CSV")
        self.neuro_export_csv_btn.clicked.connect(self._export_coords_csv)
        self.neuro_export_csv_btn.setEnabled(False)
        export_row.addWidget(self.neuro_export_csv_btn)
        export_row.addStretch()
        layout.addLayout(export_row)

        # ── Section 2: NeuroQuery Cross-Reference ──
        layout.addWidget(SectionHeader("NeuroQuery Cross-Reference"))

        nq_ctrl = QHBoxLayout()
        self.neuro_nq_btn = QPushButton("Run NeuroQuery Analysis")
        self.neuro_nq_btn.clicked.connect(self._start_neuroquery_analysis)
        nq_ctrl.addWidget(self.neuro_nq_btn)
        self.neuro_nq_status = QLabel("No NeuroQuery analysis run yet.")
        nq_ctrl.addWidget(self.neuro_nq_status, 1)
        layout.addLayout(nq_ctrl)

        # Convergence results table
        self.neuro_convergence_table = QTableWidget(0, 5)
        self.neuro_convergence_table.setHorizontalHeaderLabels([
            "Component", "Convergence Score", "Matched", "Predicted-Only",
            "Observed-Only",
        ])
        self.neuro_convergence_table.horizontalHeader().setStretchLastSection(True)
        self.neuro_convergence_table.setAlternatingRowColors(True)
        self.neuro_convergence_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.neuro_convergence_table.currentCellChanged.connect(
            self._on_convergence_row_selected
        )
        layout.addWidget(self.neuro_convergence_table)

        # Detail panel for selected component
        self.neuro_convergence_detail = QTextEdit()
        self.neuro_convergence_detail.setReadOnly(True)
        self.neuro_convergence_detail.setAccessibleName("Convergence details")
        self.neuro_convergence_detail.setMaximumHeight(120)
        self.neuro_convergence_detail.setPlaceholderText(
            "Select a component row to see matched/unmatched region details."
        )
        layout.addWidget(self.neuro_convergence_detail)

        # ── Section 3: Network Visualization ──
        layout.addWidget(SectionHeader("Condition\u2013Region Network"))

        self.neuro_network_widget = NetworkGraphWidget()
        self.neuro_network_widget.setMinimumHeight(350)
        layout.addWidget(self.neuro_network_widget, 1)

        net_ctrl = QHBoxLayout()
        self.neuro_load_network_btn = QPushButton("Load Network Data")
        self.neuro_load_network_btn.clicked.connect(self._load_network_data)
        net_ctrl.addWidget(self.neuro_load_network_btn)
        self.neuro_export_graph_btn = QPushButton("Export Graph")
        self.neuro_export_graph_btn.clicked.connect(self._export_network_graph)
        self.neuro_export_graph_btn.setEnabled(False)
        net_ctrl.addWidget(self.neuro_export_graph_btn)
        net_ctrl.addStretch()
        layout.addLayout(net_ctrl)

        scroll.setWidget(inner)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

        self._add_help_button(tab, "neuroimaging")

        # Internal state
        self._neuro_coordinates: list = []
        self._neuro_convergence_report: dict = {}

        return tab

    # ── Neuroimaging handlers ──────────────────────────────────────────

    def _start_coordinate_extraction(self):
        """Extract brain coordinates from fulltext corpus."""
        from process.coordinate_extractor import (
            extract_coordinates_from_corpus,
            convert_tal_to_mni,
        )

        self.neuro_extract_status.setText("Extracting coordinates...")
        self.neuro_extract_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            coords = extract_coordinates_from_corpus(self.pipeline_config)
            coords = convert_tal_to_mni(coords)
            self._neuro_coordinates = coords

            # Populate table
            self.neuro_coord_table.setRowCount(len(coords))
            for row, c in enumerate(coords):
                self.neuro_coord_table.setItem(row, 0, QTableWidgetItem(c.article_id))
                self.neuro_coord_table.setItem(row, 1, QTableWidgetItem(str(c.x)))
                self.neuro_coord_table.setItem(row, 2, QTableWidgetItem(str(c.y)))
                self.neuro_coord_table.setItem(row, 3, QTableWidgetItem(str(c.z)))
                self.neuro_coord_table.setItem(row, 4, QTableWidgetItem(c.space))
                self.neuro_coord_table.setItem(row, 5, QTableWidgetItem(c.region_label))
                stat = f"{c.statistic_type}={c.statistic_value}" if c.statistic_type else ""
                self.neuro_coord_table.setItem(row, 6, QTableWidgetItem(stat))
                self.neuro_coord_table.setItem(row, 7, QTableWidgetItem(str(c.n_subjects)))

            n = len(coords)
            articles = len({c.article_id for c in coords})
            self.neuro_extract_status.setText(
                f"Extracted {n} coordinates from {articles} articles."
            )
            has_coords = n > 0
            self.neuro_export_nimare_btn.setEnabled(has_coords)
            self.neuro_export_ale_btn.setEnabled(has_coords)
            self.neuro_export_csv_btn.setEnabled(has_coords)
            self._log(f"Coordinate extraction: {n} peaks from {articles} articles")

        except Exception as exc:
            self.neuro_extract_status.setText(f"Error: {exc}")
            self._log(f"Coordinate extraction failed: {exc}", "ERROR")
        finally:
            self.neuro_extract_btn.setEnabled(True)

    def _export_nimare(self):
        from process.coordinate_extractor import export_nimare_dataset

        path, _ = QFileDialog.getSaveFileName(
            self, "Export NiMARE Dataset", "nimare_dataset.json",
            "JSON files (*.json);;All files (*)",
        )
        if path:
            export_nimare_dataset(self._neuro_coordinates, Path(path))
            self._log(f"NiMARE dataset exported to {path}")

    def _export_ginger_ale(self):
        from process.coordinate_extractor import export_sleuth_format

        path, _ = QFileDialog.getSaveFileName(
            self, "Export GingerALE/Sleuth Format", "coordinates.txt",
            "Text files (*.txt);;All files (*)",
        )
        if path:
            export_sleuth_format(self._neuro_coordinates, Path(path))
            self._log(f"GingerALE/Sleuth file exported to {path}")

    def _export_coords_csv(self):
        from process.coordinate_extractor import export_csv

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Coordinates CSV", "coordinates.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            export_csv(self._neuro_coordinates, Path(path))
            self._log(f"Coordinate CSV exported to {path}")

    def _start_neuroquery_analysis(self):
        """Run NeuroQuery convergence analysis for each hypothesis component."""
        from process.neuroquery_bridge import generate_convergence_report

        self.neuro_nq_status.setText("Querying NeuroQuery API...")
        self.neuro_nq_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            # Build components list from session hypothesis components
            source_comps = (
                self.hypothesis_components
                or getattr(self.session, "hypothesis_components", [])
            )
            components = []
            for comp in source_comps:
                components.append({
                    "id": comp.id if hasattr(comp, "id") else str(comp),
                    "label": comp.label if hasattr(comp, "label") else str(comp),
                    "keywords": comp.keywords if hasattr(comp, "keywords") else [],
                })

            # Collect claims from claims directory
            claims: list[dict] = []
            claims_dir = self.pipeline_config.claims_dir
            if claims_dir.is_dir():
                import json
                for fp in claims_dir.glob("*.json"):
                    try:
                        data = json.loads(fp.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            claims.extend(data)
                        elif isinstance(data, dict):
                            claims.append(data)
                    except Exception:
                        pass

            report = generate_convergence_report(
                components, claims, config=self.pipeline_config
            )
            self._neuro_convergence_report = report

            # Populate convergence table
            rows = list(report.values())
            self.neuro_convergence_table.setRowCount(len(rows))
            for i, conv in enumerate(rows):
                label = conv.get("component_label", conv.get("component_id", ""))
                score = conv.get("convergence_score", 0.0)
                matched = len(conv.get("matched_regions", []))
                pred_only = len(conv.get("predicted_not_observed", []))
                obs_only = len(conv.get("observed_not_predicted", []))

                self.neuro_convergence_table.setItem(
                    i, 0, QTableWidgetItem(str(label))
                )
                score_item = QTableWidgetItem(f"{score:.3f}")
                if score >= 0.3:
                    score_item.setForeground(QColor(COLORS["success"]))
                elif score > 0:
                    score_item.setForeground(QColor(COLORS["warning"]))
                self.neuro_convergence_table.setItem(i, 1, score_item)
                self.neuro_convergence_table.setItem(
                    i, 2, QTableWidgetItem(str(matched))
                )
                self.neuro_convergence_table.setItem(
                    i, 3, QTableWidgetItem(str(pred_only))
                )
                self.neuro_convergence_table.setItem(
                    i, 4, QTableWidgetItem(str(obs_only))
                )

            n = len(rows)
            api_ok = sum(1 for r in rows if r.get("neuroquery_available"))
            self.neuro_nq_status.setText(
                f"Analysis complete: {n} components ({api_ok} with API data)."
            )
            self._log(f"NeuroQuery analysis: {n} components, {api_ok} with API data")

        except Exception as exc:
            self.neuro_nq_status.setText(f"Error: {exc}")
            self._log(f"NeuroQuery analysis failed: {exc}", "ERROR")
        finally:
            self.neuro_nq_btn.setEnabled(True)

    def _on_convergence_row_selected(self, row: int, col: int, prev_row: int, prev_col: int):
        """Show detail for selected convergence table row."""
        if row < 0:
            self.neuro_convergence_detail.clear()
            return
        rows = list(self._neuro_convergence_report.values())
        if row >= len(rows):
            return
        conv = rows[row]
        lines = [
            f"<b>{conv.get('component_label', '')}</b> "
            f"(score: {conv.get('convergence_score', 0):.3f})<br><br>",
        ]
        matched = conv.get("matched_regions", [])
        if matched:
            lines.append(f"<b>Matched:</b> {', '.join(matched)}<br>")
        pred = conv.get("predicted_not_observed", [])
        if pred:
            lines.append(f"<b>Predicted only:</b> {', '.join(pred)}<br>")
        obs = conv.get("observed_not_predicted", [])
        if obs:
            lines.append(f"<b>Observed only:</b> {', '.join(obs)}<br>")
        if not conv.get("neuroquery_available"):
            lines.append("<i>NeuroQuery API was unavailable for this query.</i>")
        self.neuro_convergence_detail.setHtml("".join(lines))

    def _load_network_data(self):
        """Load network analysis JSON file into the network widget."""
        import json

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Network Data", "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.neuro_network_widget.set_network_data(data)
            self.neuro_export_graph_btn.setEnabled(True)
            self._log(f"Network data loaded from {path}")
        except Exception as exc:
            self._log(f"Failed to load network data: {exc}", "ERROR")
            QMessageBox.warning(self, "Load Error", str(exc))

    def _export_network_graph(self):
        """Export the network graph as SVG or PNG."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Network Graph", "network_graph.svg",
            "SVG files (*.svg);;PNG files (*.png);;All files (*)",
        )
        if not path:
            return
        fmt = "png" if path.lower().endswith(".png") else "svg"
        self.neuro_network_widget.export_image(Path(path), fmt=fmt)
        self._log(f"Network graph exported to {path}")

    # ── Tab 11: Compare Runs ──────────────────────────────────────────

    def _build_compare_runs_tab(self) -> QWidget:
        from PySide6.QtWidgets import QListWidget

        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(SectionHeader("Stage 11: Compare Runs"))

        # ── Snapshot management section ──
        snap_group = QGroupBox("Living Review Snapshots")
        snap_layout = QVBoxLayout(snap_group)

        snap_btn_row = QHBoxLayout()
        self.snapshot_save_btn = QPushButton("Save Current Snapshot")
        self.snapshot_save_btn.clicked.connect(self._save_snapshot)
        snap_btn_row.addWidget(self.snapshot_save_btn)
        self.snapshot_refresh_btn = QPushButton("Refresh List")
        self.snapshot_refresh_btn.clicked.connect(self._refresh_snapshot_list)
        snap_btn_row.addWidget(self.snapshot_refresh_btn)
        self.snapshot_compare_btn = QPushButton("Compare Selected")
        self.snapshot_compare_btn.clicked.connect(self._compare_snapshots)
        self.snapshot_compare_btn.setEnabled(False)
        snap_btn_row.addWidget(self.snapshot_compare_btn)
        snap_btn_row.addStretch()
        snap_layout.addLayout(snap_btn_row)

        self.snapshot_status = QLabel("No snapshots loaded.")
        snap_layout.addWidget(self.snapshot_status)

        self.snapshot_list = QListWidget()
        self.snapshot_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.snapshot_list.setMaximumHeight(120)
        self.snapshot_list.itemSelectionChanged.connect(self._on_snapshot_selection_changed)
        snap_layout.addWidget(self.snapshot_list)

        layout.addWidget(snap_group)

        # ── Snapshot diff display ──
        self.snapshot_diff_display = QTextEdit()
        self.snapshot_diff_display.setReadOnly(True)
        self.snapshot_diff_display.setAccessibleName("Snapshot comparison results")
        self.snapshot_diff_display.setPlaceholderText(
            "Snapshot comparison results will appear here..."
        )
        layout.addWidget(self.snapshot_diff_display)

        # ── Manual run comparison section ──
        layout.addWidget(SectionHeader("Manual Run Comparison"))

        ra_row, self.compare_run_a, ra_btn = _make_file_row(
            "Run A:", "path/to/run_a.json"
        )
        layout.addLayout(ra_row)
        rb_row, self.compare_run_b, rb_btn = _make_file_row(
            "Run B:", "path/to/run_b.json"
        )
        layout.addLayout(rb_row)
        co_row, self.compare_output, co_btn = _make_file_row(
            "Output:", "path/to/compare.json", browse_save=True
        )
        layout.addLayout(co_row)
        json_f = "JSON files (*.json);;All files (*)"
        ra_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.compare_run_a, json_f, self.pipeline_config.claims_dir
            )
        )
        rb_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.compare_run_b, json_f, self.pipeline_config.claims_dir
            )
        )
        co_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.compare_output,
                "Markdown (*.md);;JSON files (*.json);;All files (*)",
                self.pipeline_config.output_dir,
                self.pipeline_config.data_dir / "validation",
            )
        )

        self.compare_btn = QPushButton("Run Compare Runs")
        self.compare_btn.clicked.connect(self._start_compare_runs)
        layout.addWidget(self.compare_btn)

        self.compare_progress = QProgressBar()
        self.compare_progress.setVisible(False)
        layout.addWidget(self.compare_progress)
        self.compare_status = QLabel("")
        layout.addWidget(self.compare_status)

        self.compare_results = QTextEdit()
        self.compare_results.setReadOnly(True)
        self.compare_results.setAccessibleName("Comparison results")
        layout.addWidget(self.compare_results)

        self._add_help_button(tab, "compare_runs")
        return tab

    # ── Snapshot handlers ─────────────────────────────────────────────

    def _save_snapshot(self):
        """Save current pipeline state as a snapshot."""
        from core.snapshots import save_snapshot
        try:
            path = save_snapshot(self.pipeline_config)
            self.snapshot_status.setText(f"Snapshot saved: {path.stem}")
            self._log(f"Snapshot saved: {path}")
            self._refresh_snapshot_list()
        except Exception as exc:
            self.snapshot_status.setText(f"Error: {exc}")
            self._log(f"Snapshot save failed: {exc}", level="ERROR")

    def _refresh_snapshot_list(self):
        """Reload the snapshot list from disk."""
        from core.snapshots import list_snapshots
        self.snapshot_list.clear()
        self._snapshots = list_snapshots(self.pipeline_config)
        for snap in self._snapshots:
            label = f"{snap.run_id}  ({snap.timestamp[:19]})  — {snap.summary.get('stages_completed', 0)} stages"
            self.snapshot_list.addItem(label)
        if self._snapshots:
            self.snapshot_status.setText(f"{len(self._snapshots)} snapshot(s) found.")
        else:
            self.snapshot_status.setText("No snapshots found.")
        self.snapshot_compare_btn.setEnabled(False)

    def _on_snapshot_selection_changed(self):
        """Enable compare button when exactly two snapshots are selected."""
        selected = self.snapshot_list.selectedItems()
        self.snapshot_compare_btn.setEnabled(len(selected) == 2)

    def _compare_snapshots(self):
        """Compare two selected snapshots and display the diff."""
        from core.snapshots import diff_snapshots, generate_diff_report
        selected = self.snapshot_list.selectedItems()
        if len(selected) != 2:
            return

        indices = sorted(self.snapshot_list.row(item) for item in selected)
        if not hasattr(self, "_snapshots") or len(self._snapshots) <= max(indices):
            return

        old_snap = self._snapshots[indices[1]]  # older (later in list, sorted newest first)
        new_snap = self._snapshots[indices[0]]  # newer

        try:
            diff = diff_snapshots(old_snap, new_snap, self.pipeline_config)

            # Display summary
            lines = [diff.summary_text, ""]
            if diff.new_articles:
                lines.append(f"### New Articles ({len(diff.new_articles)})")
                for art in diff.new_articles[:20]:
                    lines.append(f"  + {art.get('title', 'Untitled')} ({art.get('year', '')})")
                if len(diff.new_articles) > 20:
                    lines.append(f"  ... and {len(diff.new_articles) - 20} more")
                lines.append("")

            if diff.removed_articles:
                lines.append(f"### Removed Articles ({len(diff.removed_articles)})")
                for art in diff.removed_articles[:20]:
                    lines.append(f"  - {art.get('title', 'Untitled')} ({art.get('year', '')})")
                if len(diff.removed_articles) > 20:
                    lines.append(f"  ... and {len(diff.removed_articles) - 20} more")
                lines.append("")

            if diff.score_changes:
                lines.append(f"### Score Changes ({len(diff.score_changes)})")
                for sc in diff.score_changes[:20]:
                    lines.append(
                        f"  ~ {sc.get('title', '')[:60]}: "
                        f"{sc.get('old_score', '?')} -> {sc.get('new_score', '?')}"
                    )
                lines.append("")

            if diff.grade_changes:
                lines.append("### GRADE Assessment Changed")
                lines.append("")

            self.snapshot_diff_display.setPlainText("\n".join(lines))
            self.snapshot_status.setText(
                f"Compared {old_snap.run_id} vs {new_snap.run_id}"
            )
            self._log(
                f"Snapshot comparison: {old_snap.run_id} vs {new_snap.run_id}"
            )

            # Save report if output dir exists
            try:
                report_path = self.pipeline_config.output_dir / "snapshot_diff_report.md"
                generate_diff_report(diff, report_path)
                self._log(f"Diff report saved: {report_path}")
            except Exception:
                pass  # Non-critical

        except Exception as exc:
            self.snapshot_diff_display.setPlainText(f"Error: {exc}")
            self._log(f"Snapshot comparison failed: {exc}", level="ERROR")

    # ── Tab 12: Report & Export ───────────────────────────────────────

    def _build_report_export_tab(self) -> QWidget:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ── Top controls ──
        ctrl_row = QHBoxLayout()
        self.prisma_generate_btn = QPushButton("Generate PRISMA Report")
        self.prisma_generate_btn.setMinimumHeight(30)
        self.prisma_generate_btn.clicked.connect(self._generate_prisma_report)
        ctrl_row.addWidget(self.prisma_generate_btn)
        self.prisma_status_label = QLabel("No report generated yet.")
        ctrl_row.addWidget(self.prisma_status_label, 1)
        layout.addLayout(ctrl_row)

        # ── Splitter: diagram (top) / checklist (bottom) ──
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Upper: PRISMA flow diagram preview
        upper = QWidget()
        u_layout = QVBoxLayout(upper)
        u_layout.setContentsMargins(0, 0, 0, 0)
        u_layout.addWidget(SectionHeader("PRISMA 2020 Flow Diagram"))

        placeholder_fig = Figure(figsize=(8, 10), dpi=100, facecolor="white")
        placeholder_ax = placeholder_fig.add_subplot(111)
        placeholder_ax.text(
            0.5, 0.5,
            "Click 'Generate PRISMA Report' to create the flow diagram",
            ha="center", va="center", fontsize=11, color="#808080",
        )
        placeholder_ax.axis("off")

        self.prisma_canvas = FigureCanvasQTAgg(placeholder_fig)
        self.prisma_canvas.setMinimumHeight(300)
        u_layout.addWidget(self.prisma_canvas)
        splitter.addWidget(upper)

        # Lower: checklist table + export
        lower = QWidget()
        l_layout = QVBoxLayout(lower)
        l_layout.setContentsMargins(0, 0, 0, 0)
        l_layout.addWidget(SectionHeader("PRISMA Checklist"))

        self.prisma_checklist_table = QTableWidget(0, 5)
        self.prisma_checklist_table.setHorizontalHeaderLabels(
            ["#", "Section", "Description", "Status", "Content"]
        )
        header = self.prisma_checklist_table.horizontalHeader()
        header.setStretchLastSection(True)
        self.prisma_checklist_table.setColumnWidth(0, 32)
        self.prisma_checklist_table.setColumnWidth(1, 110)
        self.prisma_checklist_table.setColumnWidth(2, 280)
        self.prisma_checklist_table.setColumnWidth(3, 60)
        self.prisma_checklist_table.setMinimumHeight(150)
        l_layout.addWidget(self.prisma_checklist_table)

        export_scroll = QScrollArea()
        export_scroll.setWidgetResizable(True)
        export_scroll.setMinimumHeight(210)
        export_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        export_container = QWidget()
        exports_layout = QVBoxLayout(export_container)
        exports_layout.setContentsMargins(4, 4, 4, 4)
        exports_layout.setSpacing(8)

        # Export section — PRISMA
        export_group = QGroupBox("Export PRISMA Report")
        eg_layout = QHBoxLayout(export_group)
        self.prisma_export_btn = QPushButton("Export PRISMA Report...")
        self.prisma_export_btn.setEnabled(False)
        self.prisma_export_btn.setMinimumSize(210, 30)
        self.prisma_export_btn.clicked.connect(self._export_prisma_report)
        eg_layout.addWidget(self.prisma_export_btn)
        prisma_export_note = QLabel("Saves flow diagram + checklist to a directory")
        prisma_export_note.setWordWrap(True)
        eg_layout.addWidget(prisma_export_note, 1)
        eg_layout.addStretch()
        export_group.setMinimumHeight(58)
        exports_layout.addWidget(export_group)

        # Export section — Bibliography
        bib_group = QGroupBox("Export Bibliography")
        bg_layout = QHBoxLayout(bib_group)
        lbl_format = QLabel("Format:")
        bg_layout.addWidget(lbl_format)
        self.bib_format_combo = QComboBox()
        self.bib_format_combo.addItems(["BibTeX (.bib)", "RIS (.ris)", "EndNote XML (.xml)"])
        self.bib_format_combo.setMinimumWidth(180)
        lbl_format.setBuddy(self.bib_format_combo)
        bg_layout.addWidget(self.bib_format_combo)
        self.bib_export_btn = QPushButton("Export Bibliography...")
        self.bib_export_btn.setMinimumSize(210, 30)
        self.bib_export_btn.clicked.connect(self._export_bibliography)
        bg_layout.addWidget(self.bib_export_btn)
        bg_layout.addStretch()
        bib_group.setMinimumHeight(58)
        exports_layout.addWidget(bib_group)

        # Export section — Full Report Bundle
        bundle_group = QGroupBox("Full Report Bundle")
        bu_layout = QHBoxLayout(bundle_group)
        self.bundle_export_btn = QPushButton("Export Full Report Bundle...")
        self.bundle_export_btn.setMinimumSize(210, 30)
        self.bundle_export_btn.clicked.connect(self._export_report_bundle)
        bu_layout.addWidget(self.bundle_export_btn)
        self.bundle_status_label = QLabel(
            "One-click export of all pipeline outputs (PRISMA, GRADE, plots, bibliography)"
        )
        self.bundle_status_label.setWordWrap(True)
        bu_layout.addWidget(self.bundle_status_label, 1)
        bundle_group.setMinimumHeight(58)
        exports_layout.addWidget(bundle_group)

        # Export section — committee brief
        abct_group = QGroupBox("Committee Brief")
        abct_layout = QHBoxLayout(abct_group)
        self.abct_brief_export_btn = QPushButton("Export Committee Brief...")
        self.abct_brief_export_btn.setMinimumSize(210, 30)
        self.abct_brief_export_btn.clicked.connect(self._export_abct_committee_brief)
        abct_layout.addWidget(self.abct_brief_export_btn)
        self.abct_brief_status_label = QLabel(
            "Committee-facing evidence brief pack (audience and questions from "
            "the committee_brief config section)"
        )
        self.abct_brief_status_label.setWordWrap(True)
        abct_layout.addWidget(self.abct_brief_status_label, 1)
        abct_group.setMinimumHeight(58)
        exports_layout.addWidget(abct_group)

        # Export section — committee info-sheet
        abct_info_group = QGroupBox("Committee Info-Sheet")
        abct_info_layout = QHBoxLayout(abct_info_group)
        self.abct_infosheet_export_btn = QPushButton("Export Info-Sheet...")
        self.abct_infosheet_export_btn.setMinimumSize(210, 30)
        self.abct_infosheet_export_btn.clicked.connect(self._export_abct_tech_ai_infosheet)
        abct_info_layout.addWidget(self.abct_infosheet_export_btn)
        self.abct_infosheet_status_label = QLabel(
            "One-page handout (content from the abct_tech_ai_infosheet config section)"
        )
        self.abct_infosheet_status_label.setWordWrap(True)
        abct_info_layout.addWidget(self.abct_infosheet_status_label, 1)
        abct_info_group.setMinimumHeight(58)
        exports_layout.addWidget(abct_info_group)

        exports_layout.addStretch()
        export_scroll.setWidget(export_container)
        l_layout.addWidget(export_scroll)

        lower.setMinimumHeight(400)
        splitter.addWidget(lower)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([320, 480])

        layout.addWidget(splitter)
        self._add_help_button(tab, "report_export")
        return tab

    # ── PRISMA report handlers ────────────────────────────────────────

    def _generate_prisma_report(self):
        """Collect pipeline data and render the PRISMA diagram + checklist."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from process.prisma import (
            collect_prisma_data,
            generate_flow_diagram_figure,
            generate_checklist,
        )

        self.prisma_status_label.setText("Generating...")
        QApplication.processEvents()

        try:
            self._prisma_data = collect_prisma_data(
                self.pipeline_config,
                interrater_provider=self.interrater_provider.currentText(),
                interrater_model=self.interrater_model.text().strip(),
            )
            fig = generate_flow_diagram_figure(self._prisma_data)

            # Replace the canvas figure
            self.prisma_canvas.figure = fig
            fig.set_canvas(self.prisma_canvas)
            self.prisma_canvas.draw()

            # Populate checklist table
            self._prisma_checklist = generate_checklist(self._prisma_data)
            self._populate_checklist_table(self._prisma_checklist)

            self.prisma_export_btn.setEnabled(True)
            self.prisma_status_label.setText(
                f"Report generated — {self._prisma_data.records_identified} records identified, "
                f"{self._prisma_data.studies_in_review} included."
            )
            self._log("PRISMA report generated successfully.")
        except Exception as exc:
            self.prisma_status_label.setText(f"Error: {exc}")
            self._log(f"PRISMA report generation failed: {exc}", level="ERROR")
            logger.exception("PRISMA report generation failed")

    def _populate_checklist_table(self, checklist: list):
        """Fill the checklist QTableWidget from checklist data."""
        table = self.prisma_checklist_table
        table.setRowCount(len(checklist))
        for row, item in enumerate(checklist):
            # Item number
            num_item = QTableWidgetItem(str(item["item"]))
            num_item.setFlags(num_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, num_item)

            # Section
            section_text = item["section"]
            if item.get("prisma_traice"):
                section_text += " (trAIce)"
            sec_item = QTableWidgetItem(section_text)
            sec_item.setFlags(sec_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 1, sec_item)

            # Description
            desc_item = QTableWidgetItem(item["description"])
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 2, desc_item)

            # Status
            status_item = QTableWidgetItem(item["status"])
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 3, status_item)

            # Content — editable for "manual" items
            content_item = QTableWidgetItem(item["content"])
            if item["status"] == "auto":
                content_item.setFlags(content_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 4, content_item)

    def _export_prisma_report(self):
        """Export the PRISMA flow diagram and checklist to a user-selected directory."""
        from process.prisma import (
            generate_flow_diagram,
            save_checklist_markdown,
            save_checklist_csv,
        )

        directory = QFileDialog.getExistingDirectory(
            self, "Select Export Directory"
        )
        if not directory:
            return

        out = Path(directory)

        try:
            # Read back any manual edits from the table
            checklist = self._read_checklist_from_table()

            # Save flow diagram in multiple formats
            generate_flow_diagram(self._prisma_data, out / "prisma_flow_diagram", fmt="png")
            generate_flow_diagram(self._prisma_data, out / "prisma_flow_diagram", fmt="svg")

            # Save checklist
            save_checklist_markdown(checklist, out / "prisma_checklist.md")
            save_checklist_csv(checklist, out / "prisma_checklist.csv")

            self.prisma_status_label.setText(f"Exported to {directory}")
            self._log(f"PRISMA report exported to {directory}")
            QMessageBox.information(
                self, "Export Complete",
                f"PRISMA report exported to:\n{directory}\n\n"
                "Files: prisma_flow_diagram.png, prisma_flow_diagram.svg, "
                "prisma_checklist.md, prisma_checklist.csv"
            )
        except Exception as exc:
            self.prisma_status_label.setText(f"Export error: {exc}")
            self._log(f"PRISMA export failed: {exc}", level="ERROR")
            logger.exception("PRISMA export failed")

    def _read_checklist_from_table(self) -> list[dict]:
        """Read checklist data back from the table, capturing manual edits."""
        checklist = []
        table = self.prisma_checklist_table
        for row in range(table.rowCount()):
            item_widget = table.item(row, 0)
            if item_widget is None:
                continue
            section_item = table.item(row, 1)
            section_text = section_item.text() if section_item is not None else ""
            is_traice = "(trAIce)" in section_text
            section_text = section_text.replace(" (trAIce)", "")
            description_item = table.item(row, 2)
            status_item = table.item(row, 3)
            content_item = table.item(row, 4)
            checklist.append({
                "item": int(item_widget.text()) if item_widget.text().isdigit() else 0,
                "section": section_text,
                "description": description_item.text() if description_item is not None else "",
                "status": status_item.text() if status_item is not None else "",
                "content": content_item.text() if content_item is not None else "",
                "prisma_traice": is_traice,
            })
        return checklist

    # ── Bibliography & Bundle export handlers ─────────────────────────

    def _export_bibliography(self):
        """Export the corpus bibliography in the selected format."""
        from process.export import (
            export_bibtex, export_ris, export_endnote_xml,
            load_corpus_for_export,
        )

        fmt_text = self.bib_format_combo.currentText()
        if "BibTeX" in fmt_text:
            ext, export_fn = ".bib", export_bibtex
            filt = "BibTeX files (*.bib);;All files (*)"
        elif "RIS" in fmt_text:
            ext, export_fn = ".ris", export_ris
            filt = "RIS files (*.ris);;All files (*)"
        else:
            ext, export_fn = ".xml", export_endnote_xml
            filt = "XML files (*.xml);;All files (*)"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Bibliography", f"corpus{ext}", filt
        )
        if not path:
            return

        try:
            corpus = load_corpus_for_export(self.pipeline_config)
            if not corpus:
                QMessageBox.warning(
                    self, "No Data",
                    "No corpus data found. Run the Search stage first."
                )
                return
            out = export_fn(corpus, Path(path))
            self._log(f"Bibliography exported to {out}")
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(corpus)} entries to:\n{out}"
            )
        except Exception as exc:
            self._log(f"Bibliography export failed: {exc}", level="ERROR")
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_report_bundle(self):
        """Export the full report bundle to a user-selected directory."""
        from process.export import export_report_bundle

        directory = QFileDialog.getExistingDirectory(
            self, "Select Bundle Export Directory"
        )
        if not directory:
            return

        try:
            self.bundle_status_label.setText("Exporting...")
            QApplication.processEvents()
            out = export_report_bundle(self.pipeline_config, Path(directory))
            self.bundle_status_label.setText(f"Bundle exported to {out}")
            self._log(f"Report bundle exported to {out}")
            QMessageBox.information(
                self, "Bundle Complete",
                f"Report bundle exported to:\n{out}\n\n"
                "See README.md in the directory for file descriptions."
            )
        except Exception as exc:
            self.bundle_status_label.setText(f"Error: {exc}")
            self._log(f"Report bundle export failed: {exc}", level="ERROR")
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_abct_committee_brief(self):
        """Export the committee evidence brief pack."""
        from process.committee_brief import export_abct_committee_brief

        directory = QFileDialog.getExistingDirectory(
            self, "Select Committee Brief Export Directory"
        )
        if not directory:
            return

        try:
            self.abct_brief_status_label.setText("Exporting...")
            QApplication.processEvents()
            out = export_abct_committee_brief(self.pipeline_config, Path(directory))
            self.abct_brief_status_label.setText(f"Committee brief exported to {out}")
            self._log(f"Committee brief exported to {out}")
            QMessageBox.information(
                self, "Committee Brief Complete",
                f"Committee brief exported to:\n{out}\n\n"
                "See README.md in the directory for file descriptions."
            )
        except Exception as exc:
            self.abct_brief_status_label.setText(f"Error: {exc}")
            self._log(f"Committee brief export failed: {exc}", level="ERROR")
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_abct_tech_ai_infosheet(self):
        """Export the one-page committee info-sheet pack."""
        from process.committee_brief import export_abct_tech_ai_infosheet

        directory = QFileDialog.getExistingDirectory(
            self, "Select Info-Sheet Export Directory"
        )
        if not directory:
            return

        try:
            self.abct_infosheet_status_label.setText("Exporting...")
            QApplication.processEvents()
            out = export_abct_tech_ai_infosheet(self.pipeline_config, Path(directory))
            self.abct_infosheet_status_label.setText(f"Info-sheet exported to {out}")
            self._log(f"Info-sheet exported to {out}")
            QMessageBox.information(
                self, "Info-Sheet Complete",
                f"Info-sheet exported to:\n{out}\n\n"
                "The Markdown file named in the pack README is the one-page handout."
            )
        except Exception as exc:
            self.abct_infosheet_status_label.setText(f"Error: {exc}")
            self._log(f"Info-sheet export failed: {exc}", level="ERROR")
            QMessageBox.critical(self, "Export Error", str(exc))

    # ── Tab 13: Log ───────────────────────────────────────────────────

    # ═══════════════════════════════════════════════════════════════
    # Feature A: Pipeline Board
    # ═══════════════════════════════════════════════════════════════

    def _show_pipeline_board(self) -> None:
        """Create (or re-show) the floating Pipeline Status Board dialog."""
        self._refresh_pipeline_progress_from_disk(self.pipeline_config)
        if self._pipeline_board is None:
            self._pipeline_board = PipelineBoardDialog(parent=self)
            self._pipeline_board.run_all_requested.connect(self._run_all_stages)
            self._pipeline_board.stop_requested.connect(self._cancel_pipeline_stage)
            self._pipeline_board.reset_requested.connect(
                self._reset_pipeline_progress_snapshot
            )
        self._pipeline_board.apply_progress(self._pipeline_progress)
        self._pipeline_board.show()
        self._pipeline_board.raise_()
        self._pipeline_board.activateWindow()

    def _pipeline_error_summary(self, err: object) -> str:
        text = str(err).strip().replace("\n", " ")
        if len(text) > 80:
            text = text[:77] + "..."
        return text or "Failed"

    def _set_pipeline_stage(
        self,
        num: int,
        status: str,
        summary: Optional[str] = None,
    ) -> None:
        """Update the MainWindow snapshot and the board dialog if it exists."""
        if not getattr(self, "_pipeline_progress", None):
            self._pipeline_progress = empty_stage_progress()
        entry = self._pipeline_progress.setdefault(
            num, {"status": "pending", "summary": PENDING_SUMMARY}
        )
        if status == "error":
            status = "failed"
        entry["status"] = status
        if summary is not None:
            entry["summary"] = summary
        elif status == "pending":
            entry["summary"] = PENDING_SUMMARY
        elif status == "running" and not entry.get("summary"):
            entry["summary"] = "In progress…"
        board = getattr(self, "_pipeline_board", None)
        if board is not None:
            board.update_stage(num, status, summary=entry.get("summary"))

    def _refresh_pipeline_progress_from_disk(self, cfg) -> None:
        """Merge on-disk artifacts into the snapshot; skip running stages."""
        current = getattr(self, "_pipeline_progress", None) or empty_stage_progress()
        try:
            disk = collect_stage_progress(cfg)
        except Exception:
            logger.exception("Could not collect pipeline board progress from disk")
            return
        self._pipeline_progress = merge_disk_progress(current, disk)
        board = getattr(self, "_pipeline_board", None)
        if board is not None:
            board.apply_progress(self._pipeline_progress)

    def _reset_pipeline_progress_snapshot(self) -> None:
        self._pipeline_progress = empty_stage_progress()

    def _run_all_stages(self) -> None:
        """Run stages 1→10 sequentially by chaining finished signals."""
        self._log("═" * 60)
        self._log("RUN ALL STAGES requested")
        self._log("═" * 60)
        # Cancel any running worker to avoid collisions on the shared slots.
        if self._pipeline_worker is not None and self._pipeline_worker.isRunning():
            self._pipeline_worker.cancel()
        if self._search_worker is not None and self._search_worker.isRunning():
            self._search_worker.cancel()
        self._run_all_queue = [
            "search", "score", "extract", "validate", "audit",
            "synthesize", "fulltext", "interrater", "robustness", "verify",
        ]
        self._run_all_active = True
        for i in range(1, 11):
            self._set_pipeline_stage(i, "pending")
        self._run_all_stages_advance()

    def _pipeline_stage_guard(self, stage_name: str) -> bool:
        """Block a new pipeline stage if another worker is still running.

        Prevents the single `_pipeline_worker` / `_search_worker` slots from
        being overwritten and orphaning an in-progress stage.
        """
        if self._pipeline_worker is not None and self._pipeline_worker.isRunning():
            self._log(
                f"Cannot start {stage_name}: another pipeline stage is running. "
                "Wait for it to finish or click Cancel.",
                "WARNING",
            )
            return False
        if self._search_worker is not None and self._search_worker.isRunning():
            self._log(
                f"Cannot start {stage_name}: Stage 1 search is still running. "
                "Wait for it to finish or click Cancel.",
                "WARNING",
            )
            return False
        return True

    def _run_all_stages_advance(self) -> None:
        """Pop the next queued stage and start it; abort if it fails to start."""
        if not self._run_all_active:
            return
        if not self._run_all_queue:
            self._run_all_active = False
            self._log("═" * 60)
            self._log("RUN ALL STAGES complete")
            self._log("═" * 60)
            return

        stage = self._run_all_queue.pop(0)
        stage_num_map = {
            "search": 1, "score": 2, "extract": 3, "validate": 4, "audit": 5,
            "synthesize": 6, "fulltext": 7, "interrater": 8,
            "robustness": 9, "verify": 10,
        }
        stage_start_map = {
            "search": self._start_sysys_search,
            "score": self._start_sysys_score,
            "extract": self._start_extract,
            "validate": self._start_validate,
            "audit": self._start_audit,
            "synthesize": self._start_synthesize,
            "fulltext": self._start_sysys_fulltext,
            "interrater": self._start_interrater,
            "robustness": self._start_robustness,
            "verify": self._start_verify,
        }
        num = stage_num_map.get(stage, 0)
        self._set_pipeline_stage(num, "running")
        self._log(f"RUN ALL: starting stage {num} ({stage})")

        start_fn = stage_start_map.get(stage)
        if start_fn is None:
            self._log(f"RUN ALL: no start method for stage {stage}; aborting.", "ERROR")
            self._run_all_active = False
            self._run_all_queue.clear()
            return

        start_fn()

        # Detect whether a worker was actually started. Search uses its own
        # worker slot; all other pipeline stages share _pipeline_worker.
        worker = self._search_worker if stage == "search" else self._pipeline_worker
        if worker is None or not worker.isRunning():
            self._log(
                f"RUN ALL: stage {num} ({stage}) did not start; aborting chain.",
                "ERROR",
            )
            self._run_all_active = False
            self._run_all_queue.clear()
            self._set_pipeline_stage(num, "failed", summary="Did not start")

    # ═══════════════════════════════════════════════════════════════
    # Feature B: Corpus Browser Tab
    # ═══════════════════════════════════════════════════════════════

    def _build_corpus_tab(self) -> QWidget:
        """Build the Corpus Browser tab (Tab 13)."""
        import webbrowser

        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Filter bar
        filter_row = QHBoxLayout()
        lbl_corpus_filter = QLabel("Filter:")
        filter_row.addWidget(lbl_corpus_filter)
        self._corpus_filter = QLineEdit()
        self._corpus_filter.setPlaceholderText("Filter by title, author, year…")
        self._corpus_filter.setAccessibleName("Corpus filter")
        self._corpus_filter.textChanged.connect(self._filter_corpus_table)
        lbl_corpus_filter.setBuddy(self._corpus_filter)
        filter_row.addWidget(self._corpus_filter, 1)

        lbl_relevance_filter = QLabel("Rated:")
        filter_row.addWidget(lbl_relevance_filter)
        self._corpus_relevance_filter = QComboBox()
        self._corpus_relevance_filter.addItem("All papers", "all")
        self._corpus_relevance_filter.addItem("Stage 2 relevant", "stage2")
        self._corpus_relevance_filter.currentIndexChanged.connect(
            lambda _: self._filter_corpus_table(self._corpus_filter.text())
        )
        lbl_relevance_filter.setBuddy(self._corpus_relevance_filter)
        filter_row.addWidget(self._corpus_relevance_filter)

        open_doi_btn = QPushButton("Open DOI")
        open_doi_btn.setToolTip("Open the selected paper's DOI in the browser")
        open_doi_btn.clicked.connect(self._open_corpus_doi)
        filter_row.addWidget(open_doi_btn)

        export_btn = QPushButton("Export Visible")
        export_btn.setToolTip("Export visible rows to CSV")
        export_btn.clicked.connect(self._export_corpus_visible)
        filter_row.addWidget(export_btn)

        layout.addLayout(filter_row)

        # Table — virtualized model/view so large corpora never materialize a
        # widget per cell (the old QTableWidget created 7 items per paper).
        from gui.corpus_table_model import CorpusFilterProxyModel, CorpusTableModel

        self._corpus_model = CorpusTableModel(self)
        self._corpus_proxy = CorpusFilterProxyModel(self)
        self._corpus_proxy.setSourceModel(self._corpus_model)
        self._corpus_table = QTableView()
        self._corpus_table.setModel(self._corpus_proxy)
        self._corpus_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._corpus_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._corpus_table.setSortingEnabled(True)
        hh = self._corpus_table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self._corpus_table.verticalHeader().setVisible(False)
        self._corpus_table.doubleClicked.connect(self._show_corpus_abstract)
        layout.addWidget(self._corpus_table, 1)

        # Abstract preview pane
        layout.addWidget(QLabel("Abstract:"))
        self._corpus_abstract = QPlainTextEdit()
        self._corpus_abstract.setReadOnly(True)
        self._corpus_abstract.setAccessibleName("Corpus abstract preview")
        self._corpus_abstract.setMaximumHeight(160)
        self._corpus_abstract.setPlaceholderText("Double-click a row to view the abstract…")
        layout.addWidget(self._corpus_abstract)

        # Internal corpus cache for double-click lookup
        self._corpus_data: list[dict] = []
        self._corpus_stage_relevance: dict[str, set[str]] = {}
        self._corpus_stage_records: dict[str, dict[str, dict]] = {}

        return tab

    def _corpus_paper_key(self, paper: dict) -> str:
        for field in ("doi", "pmid", "s2_paper_id", "paperId", "openalex_id"):
            value = str(paper.get(field, "") or "").strip().lower()
            if value:
                return f"{field}:{value}"
        title = str(paper.get("title", "") or "").strip().lower()
        year = str(paper.get("year", "") or "").strip()
        return f"title:{title}|year:{year}"

    def _set_corpus_stage_relevance(self, stage_key: str, papers: list[dict]) -> None:
        records = {
            self._corpus_paper_key(paper): paper
            for paper in papers
            if isinstance(paper, dict)
        }
        self._corpus_stage_relevance[stage_key] = set(records)
        self._corpus_stage_records[stage_key] = records
        if self._corpus_data:
            self._populate_corpus_browser(self._corpus_data)
            self._filter_corpus_table(self._corpus_filter.text())

    def _populate_corpus_browser(self, corpus: list[dict]) -> None:
        """Fill the corpus browser table from a list of article dicts."""
        self._corpus_data = corpus
        self._corpus_model.set_corpus(
            corpus,
            self._corpus_stage_records.get("stage2", {}),
            self._corpus_paper_key,
        )
        self._corpus_table.resizeColumnToContents(0)

    def _filter_corpus_table(self, text: str) -> None:
        """Show/hide corpus table rows based on filter text."""
        stage_filter = "all"
        if hasattr(self, "_corpus_relevance_filter"):
            stage_filter = self._corpus_relevance_filter.currentData() or "all"
        if stage_filter == "all":
            self._corpus_proxy.set_stage_keys(None)
        else:
            self._corpus_proxy.set_stage_keys(
                self._corpus_stage_relevance.get(stage_filter, set())
            )
        self._corpus_proxy.set_text_filter(text)

    def _corpus_paper_for_proxy_row(self, proxy_row: int) -> dict | None:
        """Map a visible (proxy) row back to its paper dict."""
        proxy_index = self._corpus_proxy.index(proxy_row, 0)
        if not proxy_index.isValid():
            return None
        source_row = self._corpus_proxy.mapToSource(proxy_index).row()
        idx = self._corpus_model.corpus_index(source_row)
        if 0 <= idx < len(self._corpus_data):
            return self._corpus_data[idx]
        return None

    def _show_corpus_abstract(self, index) -> None:
        """Show the abstract for the double-clicked row."""
        paper = self._corpus_paper_for_proxy_row(index.row())
        if paper is not None:
            abstract = paper.get("abstract", "(no abstract available)")
            title = paper.get("title", "")
            self._corpus_abstract.setPlainText(f"{title}\n\n{abstract}")

    def _open_corpus_doi(self) -> None:
        """Open the DOI of the selected corpus row in the browser."""
        import webbrowser
        current = self._corpus_table.currentIndex()
        if not current.isValid():
            return
        paper = self._corpus_paper_for_proxy_row(current.row())
        doi = str((paper or {}).get("doi", "") or "")
        if doi:
            url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
            webbrowser.open(url)

    def _export_corpus_visible(self) -> None:
        """Export currently visible corpus rows to CSV."""
        import csv
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Corpus to CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["#", "Title", "Authors", "Year", "Source", "Score", "DOI"])
            for row in range(self._corpus_proxy.rowCount()):
                writer.writerow([
                    self._corpus_proxy.index(row, c).data() or ""
                    for c in range(7)
                ])
        self._log(f"Corpus exported to {path}")

    # ═══════════════════════════════════════════════════════════════
    # Feature C: Project System
    # ═══════════════════════════════════════════════════════════════

    def _populate_recent_menu(self) -> None:
        """Rebuild the Recent Projects submenu from QSettings."""
        self._recent_menu.clear()
        recents = ProjectManager.get_recent_projects()
        if not recents:
            no_action = QAction("(none)", self)
            no_action.setEnabled(False)
            self._recent_menu.addAction(no_action)
            return
        for path in recents:
            action = QAction(str(path), self)
            action.triggered.connect(lambda checked, p=path: self._open_project_path(p))
            self._recent_menu.addAction(action)

    def _prepare_for_new_project(self) -> None:
        if self._project_path is not None:
            self._original_question_state = OriginalQuestionState()

    def _new_project(self) -> None:
        """File > New Project..."""
        path = ProjectManager.new_project(self)
        if path:
            self._prepare_for_new_project()
            self._set_project_path(path)
            self._populate_recent_menu()
            # Try to auto-load config from new project
            cfg_yaml = path / "research_config.yaml"
            if cfg_yaml.exists():
                try:
                    self._load_research_config(cfg_yaml)
                except Exception:
                    pass
            self.statusBar().showMessage(f"New project created: {path.name}", 4000)
            self._maybe_prompt_auto_backup("new_project")

    def _open_project(self) -> None:
        """File > Open Project..."""
        path = ProjectManager.open_project(self)
        if path:
            self._open_project_path(path)

    def _open_project_path(self, path: Path) -> None:
        """Common handler for opening a project (from menu or recent list)."""
        self._original_question_state = OriginalQuestionState()
        self._set_project_path(path)
        self._populate_recent_menu()
        # Restore state
        state = ProjectManager.load_project(path)
        cfg_yaml = path / "research_config.yaml"
        if cfg_yaml.exists():
            try:
                self._load_research_config(cfg_yaml)
                self.statusBar().showMessage(f"Loaded config from {cfg_yaml.name}", 4000)
            except Exception as exc:
                self._log(f"Could not load config: {exc}", "WARNING")
        from core.domain_config import restore_domain_config
        restore_domain_config(self.pipeline_config, state.get("domain_config"))
        if self._restore_fulltext_project_config(state):
            self._sync_settings_to_tabs()
        # Restore hypothesis components if saved
        window_state = state.get("window_state", {})
        self._original_question_state = OriginalQuestionState.from_window_state(
            window_state
        )
        saved_query = str(window_state.get("search_query") or "").strip()
        if not saved_query:
            saved_query = self._original_question_state.value
        if saved_query and hasattr(self, "sysys_nl_query"):
            self.sysys_nl_query.setText(saved_query)
        saved_components = ProjectManager.deserialize_hypothesis_components(
            window_state.get("hypothesis_components", [])
        )
        if saved_components:
            self.hypothesis_components = saved_components
            self._refresh_component_list()
            self._update_search_workflow_buttons()
        saved_boolean_query = str(window_state.get("weighted_boolean_query") or "")
        if saved_boolean_query and not saved_components and hasattr(self, "boolean_query_preview"):
            self.boolean_query_preview.setPlainText(saved_boolean_query)
        saved_paper_limit = window_state.get("paper_limit")
        if saved_paper_limit is not None and hasattr(self, "paper_limit"):
            try:
                self.paper_limit.setValue(int(saved_paper_limit))
            except (TypeError, ValueError):
                pass
        if hasattr(self, "sysys_ft_resume"):
            stage7 = window_state.get("stage7", {})
            self.sysys_ft_corpus.setText(str(stage7.get("claims_path") or ""))
            self.sysys_ft_resume.setChecked(bool(stage7.get("resume", True)))
            self.sysys_ft_workers.setValue(int(stage7.get(
                "workers", self.pipeline_config.fulltext_download_workers
            )))
        self._refresh_pipeline_progress_from_disk(self.pipeline_config)
        self.statusBar().showMessage(f"Opened project: {path.name}", 4000)

    def _save_project(self) -> None:
        """File > Save Project."""
        if self._project_path is None:
            # Prompt to create/choose a folder
            path = ProjectManager.open_project(self)
            if not path:
                return
            self._set_project_path(path)
        assert self._project_path is not None
        self._sync_fulltext_controls_to_config()
        weighted_boolean_query = ""
        if hasattr(self, "boolean_query_preview"):
            weighted_boolean_query = self.boolean_query_preview.toPlainText().strip()
        window_state = {
            **self._project_question_window_state(),
            "weighted_boolean_query": weighted_boolean_query,
            "hypothesis_components": ProjectManager.serialize_hypothesis_components(
                self.hypothesis_components
            ),
            "paper_limit": self.paper_limit.value() if hasattr(self, "paper_limit") else 500,
        }
        if hasattr(self, "sysys_ft_resume"):
            window_state["stage7"] = {
                "claims_path": self.sysys_ft_corpus.text().strip(),
                "workers": self.sysys_ft_workers.value(),
                "resume": self.sysys_ft_resume.isChecked(),
            }
        ProjectManager.save_project(
            self._project_path, self.pipeline_config, window_state
        )
        self.statusBar().showMessage(
            f"Project saved to {self._project_path.name}", 4000
        )
        self._schedule_auto_backup("save")

    def _backup_project(self) -> None:
        """File > Backup Project..."""
        if self._project_path is None:
            QMessageBox.warning(
                self,
                "Backup Project",
                "Open or save a project before creating a backup.",
            )
            return

        self._save_project()
        destination = QFileDialog.getExistingDirectory(
            self,
            "Choose Backup Destination",
            str(Path.home()),
        )
        if not destination:
            return

        try:
            backup_path = ProjectManager.backup_project(
                self._project_path,
                Path(destination),
            )
            self._log(f"Project backup created: {backup_path}")
            self.statusBar().showMessage(f"Backup created: {backup_path.name}", 5000)
            QMessageBox.information(
                self,
                "Backup Complete",
                f"Project backup created:\n{backup_path}",
            )
        except Exception as exc:
            self._log(f"Backup failed: {exc}", "ERROR")
            QMessageBox.critical(self, "Backup Failed", str(exc))

    def _restore_project_backup(self) -> None:
        """File > Restore Project Backup..."""
        from PySide6.QtCore import QSettings

        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        last_backup_value = cast(str, settings.value("backup/last_path", "", type=str)) or ""
        backup_destination_value = cast(str, settings.value("backup/destination", "", type=str)) or ""
        last_backup_path = Path(last_backup_value).expanduser() if last_backup_value else None
        backup_destination = (
            Path(backup_destination_value).expanduser() if backup_destination_value else None
        )
        default_backup_dir = Path.home()
        if last_backup_path is not None and last_backup_path.exists():
            default_backup_dir = last_backup_path
        elif backup_destination is not None and backup_destination.exists():
            default_backup_dir = backup_destination

        source_choice = QMessageBox.question(
            self,
            "Restore Project Backup",
            "Restore from a .zip backup?\n\nChoose No to restore from a backup folder instead.",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if source_choice == QMessageBox.StandardButton.Cancel:
            return

        if source_choice == QMessageBox.StandardButton.Yes:
            backup, _ = QFileDialog.getOpenFileName(
                self,
                "Choose SystS Backup Zip",
                str(default_backup_dir if default_backup_dir.is_dir() else default_backup_dir.parent),
                "SystS backups (*.zip);;All files (*)",
            )
        else:
            backup = QFileDialog.getExistingDirectory(
                self,
                "Choose SystS Backup Folder",
                str(default_backup_dir),
            )
        if not backup:
            return

        try:
            metadata = ProjectManager.backup_metadata(Path(backup))
        except Exception as exc:
            self._log(f"Restore failed: {exc}", "ERROR")
            QMessageBox.critical(self, "Restore Failed", str(exc))
            return

        project_name = str(metadata.get("project_name") or Path(backup).stem or "SystS_Project")
        project_name = ProjectManager._safe_backup_name(project_name)
        source_project_value = str(metadata.get("source_project_path") or "")
        source_path = Path(source_project_value).expanduser() if source_project_value else None
        restore_parent = (
            source_path.parent
            if source_path is not None and source_path.parent.exists()
            else Path.home()
        )
        destination = QFileDialog.getExistingDirectory(
            self,
            "Choose Folder to Restore Into",
            str(restore_parent),
        )
        if not destination:
            return

        destination_path = Path(destination) / project_name
        overwrite = False
        if destination_path.exists() and any(destination_path.iterdir()):
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Restore Destination Not Empty")
            box.setText(f"{destination_path} is not empty.")
            box.setInformativeText("Overwrite it or restore to a timestamped copy?")
            overwrite_btn = box.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
            copy_btn = box.addButton("Timestamped Copy", QMessageBox.ButtonRole.AcceptRole)
            cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked == cancel_btn:
                return
            if clicked == overwrite_btn:
                overwrite = True
            elif clicked == copy_btn:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                destination_path = destination_path.parent / f"{destination_path.name}_restored_{stamp}"

        try:
            restored_path = ProjectManager.restore_project(
                Path(backup),
                destination_path,
                overwrite=overwrite,
            )
            self._open_project_path(restored_path)
            self._log(f"Project restored from backup: {restored_path}")
            QMessageBox.information(
                self,
                "Restore Complete",
                f"Project restored and opened:\n{restored_path}",
            )
        except Exception as exc:
            self._log(f"Restore failed: {exc}", "ERROR")
            QMessageBox.critical(self, "Restore Failed", str(exc))

    def _configure_auto_backup(self, default_enabled: bool | None = None) -> bool:
        """Show automatic backup options and persist them to QSettings."""
        from PySide6.QtCore import QSettings
        from gui.dialogs import AutoBackupOptionsDialog

        dialog = AutoBackupOptionsDialog(
            self,
            default_enabled=default_enabled,
            project_path=self._project_path,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False

        values = dialog.values()
        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        settings.setValue("backup/enabled", values["enabled"])
        settings.setValue("backup/prompt_disabled", not values["enabled"])
        settings.setValue("backup/destination", values["destination"])
        settings.setValue("backup/include_images", values["include_images"])
        settings.setValue("backup/include_fulltext", values["include_fulltext"])
        settings.setValue("backup/on_stage_complete", values["on_stage_complete"])
        settings.setValue("backup/on_exit", values["on_exit"])
        settings.setValue("backup/min_interval_minutes", values["min_interval_minutes"])
        settings.setValue("backup/max_file_mb", values["max_file_mb"])
        if values["enabled"]:
            self._log(f"Automatic project backup enabled: {values['destination']}")
            self.statusBar().showMessage("Automatic project backup enabled", 4000)
        else:
            self._log("Automatic project backup disabled")
            self.statusBar().showMessage("Automatic project backup disabled", 4000)
        if values["enabled"] and self._project_path is not None:
            self._schedule_auto_backup("auto-backup enabled", force=True)
        return values["enabled"]

    def _auto_backup_options(self) -> dict:
        from PySide6.QtCore import QSettings

        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        return {
            "enabled": settings.value("backup/enabled", False, type=bool),
            "destination": settings.value("backup/destination", "", type=str),
            "include_images": settings.value("backup/include_images", True, type=bool),
            "include_fulltext": settings.value("backup/include_fulltext", False, type=bool),
            "on_stage_complete": settings.value("backup/on_stage_complete", True, type=bool),
            "on_exit": settings.value("backup/on_exit", True, type=bool),
            "min_interval_minutes": cast(int, settings.value("backup/min_interval_minutes", 10, type=int)),
            "max_file_mb": cast(int, settings.value("backup/max_file_mb", 100, type=int)),
        }

    def _schedule_auto_backup(self, reason: str, force: bool = False) -> None:
        options = self._auto_backup_options()
        if not options["enabled"] or self._project_path is None:
            return
        if not force and reason != "save" and not options["on_stage_complete"]:
            return
        if self._backup_worker is not None and self._backup_worker.isRunning():
            self._pending_auto_backup_reason = reason
            return

        now = time.monotonic()
        min_interval = max(1, options["min_interval_minutes"]) * 60
        debounce_seconds = 5 if reason == "save" else 45
        if force:
            delay_ms = 0
        else:
            next_allowed = (
                self._last_auto_backup_monotonic + min_interval
                if self._last_auto_backup_monotonic > 0
                else now
            )
            delay_seconds = max(debounce_seconds, next_allowed - now)
            delay_ms = int(max(0, delay_seconds) * 1000)
        self._pending_auto_backup_reason = reason
        self._auto_backup_timer.start(delay_ms)

    def _run_auto_backup(
        self,
        reason: str = "scheduled",
        *,
        force: bool = False,
        synchronous: bool = False,
    ) -> None:
        from PySide6.QtCore import QSettings

        options = self._auto_backup_options()
        if not options["enabled"] or self._project_path is None:
            return
        if not options["destination"]:
            self._log("Auto-backup skipped: no destination configured", "WARNING")
            return
        if not force and self.pipeline_state != PipelineState.IDLE:
            self._schedule_auto_backup(reason or "pipeline active")
            return
        if self._backup_worker is not None and self._backup_worker.isRunning():
            self._pending_auto_backup_reason = reason
            return

        current_reason = self._pending_auto_backup_reason or reason
        self._pending_auto_backup_reason = ""

        backup_options = {
            "include_fulltext": options["include_fulltext"],
            "include_images": options["include_images"],
            "max_file_mb": options["max_file_mb"],
        }
        if synchronous:
            try:
                result = ProjectManager.sync_project_backup(
                    self._project_path,
                    Path(options["destination"]),
                    **backup_options,
                )
                self._record_auto_backup_result(result)
            except Exception as exc:
                self._log(f"Auto-backup failed: {exc}", "ERROR")
            return

        self._backup_worker = ProjectBackupWorker(
            self._project_path,
            Path(options["destination"]),
            **backup_options,
        )
        self._backup_worker.progress.connect(lambda msg: self._log(msg))
        self._backup_worker.finished_backup.connect(self._on_auto_backup_finished)
        self._backup_worker.error.connect(self._on_auto_backup_error)
        QSettings("Sysys", "HypothesisConsensusAnalyzer").setValue(
            "backup/last_reason",
            current_reason,
        )
        self._log(f"Auto-backup scheduled/run reason: {current_reason}")
        self._backup_worker.start()

    def _record_auto_backup_result(self, result: dict) -> None:
        from PySide6.QtCore import QSettings

        self._last_auto_backup_monotonic = time.monotonic()
        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        settings.setValue("backup/last_success", result.get("created_at", ""))
        settings.setValue("backup/last_path", result.get("backup_root", ""))
        copied = int(result.get("copied", 0) or 0)
        unchanged = int(result.get("unchanged", 0) or 0)
        self._log(
            f"Auto-backup complete: {copied} copied, {unchanged} unchanged → "
            f"{result.get('backup_root', '')}"
        )
        self.statusBar().showMessage(
            f"Auto-backup complete: {copied} changed files copied",
            5000,
        )

    def _on_auto_backup_finished(self, result: dict) -> None:
        self._record_auto_backup_result(result)
        self._backup_worker = None
        pending = self._pending_auto_backup_reason
        self._pending_auto_backup_reason = ""
        if pending:
            self._schedule_auto_backup(pending)

    def _on_auto_backup_error(self, error: str) -> None:
        self._backup_worker = None
        self._log(f"Auto-backup failed: {error}", "ERROR")
        self.statusBar().showMessage("Auto-backup failed; see log", 5000)

    def closeEvent(self, event) -> None:
        """Persist project state and optionally run a final lightweight backup."""
        try:
            if self._project_path is not None:
                self._save_project()
            options = self._auto_backup_options()
            if options["enabled"] and options["on_exit"] and self._project_path is not None:
                self._auto_backup_timer.stop()
                if self._backup_worker is not None and self._backup_worker.isRunning():
                    self._log("Waiting briefly for active auto-backup before exit...")
                    if not self._backup_worker.wait(3000):
                        self._log("Auto-backup still running; skipping exit backup.", "WARNING")
                        super().closeEvent(event)
                        return
                    self._backup_worker = None
                self._run_auto_backup(reason="exit", force=True, synchronous=True)
        except Exception as exc:
            self._log(f"Close-time backup failed: {exc}", "ERROR")
        super().closeEvent(event)

    def _set_project_path(self, path: Path) -> None:
        self._project_path = path
        self.setWindowTitle(
            f"SystS — {path.name}"
        )
        try:
            self.pipeline_config.project_root = path
            self.pipeline_config._resolve_paths()
        except Exception:
            pass
        if hasattr(self, "score_corpus_path"):
            self._sync_all_stage_paths(self.pipeline_config)

    def _load_research_config(self, cfg_path: Path) -> None:
        """Load a research_config.yaml file into pipeline_config."""
        try:
            new_cfg = type(self.pipeline_config).load_config(
                cfg_path,
                project_root=cfg_path.parent,
            )
            self.pipeline_config = new_cfg
            self._restore_saved_keys()
            self._populate_components_from_config()
            self._sync_settings_to_tabs()
            if hasattr(self, "score_corpus_path"):
                self._sync_all_stage_paths(self.pipeline_config)
        except TypeError:
            # load_config may not accept a path argument — try default
            try:
                import os
                orig = os.getcwd()
                os.chdir(cfg_path.parent)
                new_cfg = type(self.pipeline_config).load_config()
                self.pipeline_config = new_cfg
                self._restore_saved_keys()
                self._populate_components_from_config()
                self._sync_settings_to_tabs()
                os.chdir(orig)
            except Exception as exc:
                raise RuntimeError(f"Could not load config: {exc}") from exc

    # ═══════════════════════════════════════════════════════════════
    # Feature D: At-a-Glance Panel helpers
    # ═══════════════════════════════════════════════════════════════

    def _toggle_glance_panel(self) -> None:
        """Show/hide the At-a-Glance panel."""
        if self._glance_frame.isVisible():
            self._glance_frame.hide()
            self._glance_toggle_btn.setText("▼ Show")
        else:
            self._glance_frame.show()
            self._glance_toggle_btn.setText("▲ Hide")

    def _update_glance_corpus(self, n: int) -> None:
        if hasattr(self, "_card_corpus"):
            self._card_corpus.set_value(f"{n} articles")

    def _update_glance_relevant(self, n: int) -> None:
        if hasattr(self, "_card_relevant"):
            self._card_relevant.set_value(f"{n} papers")

    def _update_glance_stage(self, stage_name: str) -> None:
        if hasattr(self, "_card_stage"):
            self._card_stage.set_value(stage_name)

    # ═══════════════════════════════════════════════════════════════
    # Feature F: Tab count badges
    # ═══════════════════════════════════════════════════════════════

    def _set_tab_badge(self, idx: int, base: str, count: int | None = None) -> None:
        """Set a tab's label with an optional count badge."""
        label = f"{base} ({count})" if count is not None else base
        self.tabs.setTabText(idx, label)

    # ═══════════════════════════════════════════════════════════════
    # Feature G: Keyboard Shortcuts
    # ═══════════════════════════════════════════════════════════════

    def _setup_shortcuts(self) -> None:
        """Install Ctrl+1–9 tab switchers and Ctrl+R run-current-tab shortcut."""
        from PySide6.QtGui import QShortcut
        # Tab navigation: Ctrl+1 through Ctrl+9
        for i in range(9):
            sc = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            sc.activated.connect(lambda idx=i: self.tabs.setCurrentIndex(idx))
        # Run current tab action
        run_sc = QShortcut(QKeySequence("Ctrl+R"), self)
        run_sc.activated.connect(self._run_current_tab_action)

    def _run_current_tab_action(self) -> None:
        """Run the primary action for the currently visible tab."""
        idx = self.tabs.currentIndex()
        actions = {
            0: self._start_sysys_search,
            1: self._start_sysys_score,
            2: self._start_extract,
            3: self._start_audit,
            4: self._start_synthesize,
            5: self._start_fulltext_retrieval,
            6: self._start_robustness,
        }
        action = actions.get(idx)
        if action:
            action()

    # ═══════════════════════════════════════════════════════════════
    # Feature H: Config Auto-Load
    # ═══════════════════════════════════════════════════════════════

    def _auto_load_config(self) -> None:
        """Scan cwd (and parent) for research_config.yaml and auto-load it."""
        cwd = Path.cwd()
        for search_dir in [cwd, cwd.parent]:
            cfg_path = search_dir / "research_config.yaml"
            if cfg_path.exists():
                try:
                    self._load_research_config(cfg_path)
                    try:
                        rel = cfg_path.relative_to(cwd)
                    except ValueError:
                        rel = cfg_path
                    self.statusBar().showMessage(
                        f"Auto-loaded config from {rel}", 5000
                    )
                    return
                except Exception:
                    pass

    def _build_log_window(self):
        """Create the application Log as a floating window (opened from File menu).

        Previously the Log was a crowded tab; it now lives in its own dockable
        window (like the Output Console) toggled via File → Log Window.
        """
        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setAccessibleName("Application log")
        # Bound the document: long pipeline runs append unboundedly otherwise.
        self.log_output.document().setMaximumBlockCount(5000)
        self.log_output.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 14pt;"
        )
        layout.addWidget(self.log_output)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_log)
        save_btn = QPushButton("Save Log...")
        save_btn.clicked.connect(self._save_log)
        self.log_autoscroll = QCheckBox("Auto-scroll")
        self.log_autoscroll.setChecked(True)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(self.log_autoscroll)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._log_dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)
        # Floating, hidden by default — the user opens it from the File menu.
        self._log_dock.setFloating(True)
        self._log_dock.resize(760, 420)
        self._log_dock.hide()

        # Keep the File-menu checkbox in sync with the window's visibility.
        self._log_dock.visibilityChanged.connect(
            lambda visible: self._toggle_log_action.setChecked(visible)
        )

    def _toggle_log_window(self, checked: bool):
        """Show or hide the log window from the File menu."""
        if checked:
            self._log_dock.show()
            self._log_dock.raise_()
            self._log_dock.activateWindow()
        else:
            self._log_dock.hide()

    # ── Log helpers ───────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] [{level}] {msg}"

        # Log window (may not exist yet during early UI construction)
        if hasattr(self, "log_output"):
            self.log_output.append(line)
            if getattr(self, "log_autoscroll", None) and self.log_autoscroll.isChecked():
                sb = self.log_output.verticalScrollBar()
                sb.setValue(sb.maximum())

        # Output console (color-coded)
        if hasattr(self, '_console_output'):
            if level == "ERROR":
                color = "#ff6b6b"
            elif level == "WARNING":
                color = "#ffa94d"
            else:
                color = "#e0e0e0"
            self._console_output.append(
                f'<span style="color:{color}">{line}</span>'
            )
            csb = self._console_output.verticalScrollBar()
            csb.setValue(csb.maximum())

    def _clear_log(self):
        self.log_output.clear()

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "", "Text files (*.txt);;All files (*)"
        )
        if path:
            with open(path, "w") as f:
                f.write(self.log_output.toPlainText())
            self._log(f"Log saved to {path}")

    # ── File browse helpers ───────────────────────────────────────────

    def _browse_open_file(
        self,
        line_edit: QLineEdit,
        file_filter: str = "All files (*)",
        initial_dir: Path | None = None,
        fallback_dir: Path | None = None,
    ):
        start = ""
        for d in (initial_dir, fallback_dir, self.pipeline_config.project_root):
            if d is not None:
                try:
                    p = Path(d)
                    if p.is_dir():
                        start = str(p)
                        break
                except OSError:
                    continue
        cur = line_edit.text().strip()
        if cur:
            rp = _paths_resolve_under_root(cur, self.pipeline_config.project_root)
            if rp and rp.is_file():
                start = str(rp.parent)
        path, _ = QFileDialog.getOpenFileName(self, "Open File", start, file_filter)
        if path:
            line_edit.setText(path)

    def _browse_save_file(
        self,
        line_edit: QLineEdit,
        file_filter: str = "All files (*)",
        initial_dir: Path | None = None,
        fallback_dir: Path | None = None,
    ):
        start = ""
        for d in (initial_dir, fallback_dir, self.pipeline_config.project_root):
            if d is not None:
                try:
                    p = Path(d)
                    if p.is_dir():
                        start = str(p)
                        break
                except OSError:
                    continue
        cur = line_edit.text().strip()
        if cur:
            rp = _paths_resolve_under_root(cur, self.pipeline_config.project_root)
            if rp and rp.parent.is_dir():
                start = str(rp.parent)
            elif rp is None:
                p = Path(cur).expanduser()
                if not p.is_absolute():
                    p = self.pipeline_config.project_root / p
                if p.parent.is_dir():
                    start = str(p.parent)
        path, _ = QFileDialog.getSaveFileName(self, "Save File", start, file_filter)
        if path:
            line_edit.setText(path)

    def _browse_project_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select Project Root")
        if path:
            self.pipeline_config.project_root = Path(path)
            self.pipeline_config.__post_init__()

    # ── Hypothesis component management ───────────────────────────────

    def _refresh_component_list(self):
        from analysis.scoring import quote_boolean_term, build_weighted_boolean_query

        self.component_list.blockSignals(True)
        self.component_list.setRowCount(len(self.hypothesis_components))
        for i, comp in enumerate(self.hypothesis_components):
            self.component_list.setItem(i, 0, QTableWidgetItem(comp.id))
            self.component_list.setItem(i, 1, QTableWidgetItem(comp.label))
            self.component_list.setItem(i, 2, QTableWidgetItem(", ".join(comp.keywords)))
            self.component_list.setItem(i, 3, QTableWidgetItem(str(comp.weight)))
            terms = [quote_boolean_term(term) for term in (comp.keywords or [comp.label])]
            terms = [term for term in terms if term]
            self.component_list.setItem(i, 4, QTableWidgetItem(" OR ".join(terms[:4])))
        self.component_list.blockSignals(False)
        if hasattr(self, "boolean_query_preview"):
            query = build_weighted_boolean_query(self.hypothesis_components)
            self.boolean_query_preview.setPlainText(query)

    def _sync_query_components_from_question(self, query: str):
        """Create or refresh weighted hypothesis/query components."""
        from analysis.scoring import build_hypothesis_components

        existing = {comp.id: comp for comp in self.hypothesis_components}
        if self.stat_extract_llm_context.isChecked():
            cfg = self._apply_config_overrides_for_stage(
                self.stat_extract_provider, self.stat_extract_model)
        else:
            cfg = self.pipeline_config
        components = build_hypothesis_components(
            hypothesis=getattr(cfg, "hypothesis_text", "") or query,
            research_config=getattr(cfg, "_research_config", {}) or {},
            co_occurring_conditions=getattr(cfg, "co_occurring_conditions", []) or [],
            query=query,
        )
        for comp in components:
            old = existing.get(comp.id)
            if old is None:
                continue
            comp.weight = old.weight
            comp.description = old.description

        if components:
            self.hypothesis_components = components
            self._refresh_component_list()
            self._update_search_workflow_buttons()
            self._log(
                f"Hypothesis parts prepared for keyword weighting: {len(components)} components"
            )

    def _update_search_workflow_buttons(self):
        """Gate Search controls around the search → download → analysis workflow."""
        # Fetch Articles is a download-only action, so it needs an existing
        # corpus (in memory or a saved corpus.json), not just a question.
        has_corpus = bool(self._pipeline_corpus) or bool(self.articles)
        if not has_corpus and getattr(self, "pipeline_config", None) is not None:
            try:
                has_corpus = (self.pipeline_config.corpus_dir / "corpus.json").exists()
            except Exception:
                has_corpus = False
        if hasattr(self, "fetch_btn"):
            self.fetch_btn.setEnabled(has_corpus)
        if hasattr(self, "analysis_btn"):
            has_results = bool(self.articles or self._pipeline_corpus)
            self.analysis_btn.setEnabled(bool(self._keyword_analysis_ready and has_results))

    def _edit_component(self):
        row = self.component_list.currentRow()
        if row < 0 or row >= len(self.hypothesis_components):
            return
        from gui.dialogs import ComponentEditor
        comp = self.hypothesis_components[row]
        dlg = ComponentEditor(comp, parent=self)
        if dlg.exec():
            self._refresh_component_list()

    def _populate_components_from_config(self):
        """Build HypothesisComponent objects from pipeline_config fields."""
        from core.research_components import build_hypothesis_components_from_config

        components = build_hypothesis_components_from_config(self.pipeline_config)

        if components:
            self.hypothesis_components = components
            self._refresh_component_list()
            self._update_search_workflow_buttons()
            self._log(f"Loaded {len(components)} hypothesis components from config")

    def _reset_components(self):
        # Try loading from the current pipeline config first
        previous_count = len(self.hypothesis_components)
        self._populate_components_from_config()
        if self.hypothesis_components and len(self.hypothesis_components) != previous_count:
            return
        if self.hypothesis_components:
            self._refresh_component_list()
            return
        cfg = self.pipeline_config
        if getattr(cfg, "research_config", {}):
            self._populate_components_from_config()
            return
        self.hypothesis_components = []
        self._refresh_component_list()
        self._update_search_workflow_buttons()

    # ── Persisted keys ────────────────────────────────────────────────────

    def _restore_saved_keys(self):
        """Restore API keys and LLM settings from QSettings into pipeline_config.

        API keys only fill empty fields after Config.__post_init__ so env vars
        keep priority. LLM settings are user preferences, so saved provider/model
        choices override non-empty dataclass defaults such as OpenAI.
        """
        from PySide6.QtCore import QSettings
        from gui.settings_persistence import restore_saved_settings

        s = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        restore_saved_settings(self.pipeline_config, s)

    # ── Sync settings → tab widgets ────────────────────────────────────────

    def _sync_stage_path_line_edit(self, edit: QLineEdit, path: Path) -> None:
        """Set path when the file exists and the field is empty or stale / default."""
        try:
            exists = path.exists()
        except OSError:
            exists = False
        if not exists:
            return
        try:
            target = path.resolve()
        except OSError:
            target = path
        cur = edit.text().strip()
        if not cur:
            edit.setText(str(path))
            return
        resolved = _paths_resolve_under_root(cur, self.pipeline_config.project_root)
        if resolved is None:
            edit.setText(str(path))
            return
        try:
            cur_res = resolved.resolve()
        except OSError:
            edit.setText(str(path))
            return
        if not resolved.exists() or cur_res == target:
            edit.setText(str(path))

    def _sync_compare_run_paths(self, cfg: Config) -> None:
        """Pre-fill manual compare inputs from the current data layout."""
        claims_p = cfg.claims_dir / "claims.json"
        filt_p = cfg.claims_dir / "claims_filtered.json"
        rel_p = cfg.corpus_dir / "relevant.json"
        out_md = cfg.output_dir / "run_compare_report.md"
        out_json = cfg.data_dir / "validation" / "run_compare_results.json"

        if not self.compare_run_a.text().strip():
            if claims_p.exists():
                self.compare_run_a.setText(str(claims_p))
            elif rel_p.exists():
                self.compare_run_a.setText(str(rel_p))
        if not self.compare_run_b.text().strip():
            if filt_p.exists():
                self.compare_run_b.setText(str(filt_p))
            elif claims_p.exists():
                a_res = _paths_resolve_under_root(
                    self.compare_run_a.text(), cfg.project_root
                )
                try:
                    same = a_res and a_res.resolve() == claims_p.resolve()
                except OSError:
                    same = False
                if not same:
                    self.compare_run_b.setText(str(claims_p))
            elif rel_p.exists():
                self.compare_run_b.setText(str(rel_p))

        if not self.compare_output.text().strip():
            self.compare_output.setText(str(out_md))
        else:
            co = self.compare_output.text().strip()
            if co.endswith(".json") and out_json.exists():
                self._sync_stage_path_line_edit(self.compare_output, out_json)

    def _sync_all_stage_paths(self, cfg: Config) -> None:
        """Align every stage file path field with cfg.data_dir / standard filenames."""
        corpus_json = cfg.corpus_dir / "corpus.json"
        relevant_json = cfg.claims_dir / "relevant.json"
        claims_json = cfg.claims_dir / "claims.json"
        filtered_json = cfg.claims_dir / "claims_filtered.json"
        method_profiles = cfg.data_dir / "method_review" / "method_profiles_validated.json"
        metrics_json = cfg.data_dir / "fulltext" / "auto_extracted_metrics.json"
        robustness_json = cfg.data_dir / "validation" / "robustness_results.json"

        self._sync_stage_path_line_edit(self.score_corpus_path, corpus_json)
        if hasattr(self, "stat_extract_input"):
            self._sync_stage_path_line_edit(self.stat_extract_input, relevant_json)
            self._sync_stage_path_line_edit(
                self.stat_extract_output,
                cfg.claims_dir / "statistical_extractions.json",
            )
        self._sync_stage_path_line_edit(self.extract_corpus, relevant_json)
        self._sync_stage_path_line_edit(self.extract_output, claims_json)
        validate_in = claims_json if claims_json.exists() else relevant_json
        self._sync_stage_path_line_edit(self.validate_claims, validate_in)
        self._sync_stage_path_line_edit(self.validate_output, method_profiles)

        audit_in = claims_json if claims_json.exists() else relevant_json
        self._sync_stage_path_line_edit(self.audit_input, audit_in)
        self._sync_stage_path_line_edit(self.audit_output, filtered_json)

        synth_in = filtered_json if filtered_json.exists() else claims_json
        self._sync_stage_path_line_edit(self.synth_input, synth_in)
        self._sync_stage_path_line_edit(self.synth_output, cfg.clusters_dir / "clusters.json")

        self._sync_stage_path_line_edit(self.ft_output_path, metrics_json)
        self._sync_stage_path_line_edit(self.sysys_ft_corpus, synth_in)

        inter_in = (
            filtered_json if filtered_json.exists()
            else claims_json if claims_json.exists()
            else relevant_json
        )
        self._sync_stage_path_line_edit(self.interrater_corpus, inter_in)

        robust_in = inter_in
        self._sync_stage_path_line_edit(self.robust_input, robust_in)
        self._sync_stage_path_line_edit(self.robust_output, robustness_json)

        verify_in = (
            filtered_json if filtered_json.exists()
            else claims_json if claims_json.exists()
            else relevant_json
        )
        self._sync_stage_path_line_edit(self.verify_claims, verify_in)

        self._sync_compare_run_paths(cfg)

        # Load existing run data from disk into memory
        self._load_existing_run_data(cfg)

    def _load_existing_run_data(self, cfg) -> None:
        """Load previous pipeline outputs from disk into the GUI state.

        Checks for corpus.json, relevant.json, claims.json etc. and populates
        the articles dict, glance panel, corpus browser, and result displays.
        """
        import json as _json
        from core.models import Article

        # 1. Load corpus
        corpus_path = cfg.corpus_dir / "corpus.json"
        if corpus_path.exists():
            try:
                with open(corpus_path, encoding="utf-8") as f:
                    corpus = _json.load(f)
                if isinstance(corpus, list) and corpus:
                    self._pipeline_corpus = corpus

                    # Build articles dict
                    self.articles = {}
                    for p in corpus:
                        art_id = p.get("doi") or p.get("pmid") or p.get("title", "")[:60]
                        self.articles[art_id] = Article(
                            id=art_id,
                            title=p.get("title", ""),
                            authors=p.get("authors", []) if isinstance(p.get("authors"), list) else [],
                            abstract=p.get("abstract", ""),
                            year=p.get("year"),
                            doi=p.get("doi", ""),
                            pmid=p.get("pmid", ""),
                            citations=p.get("citationCount", 0) or p.get("citations", 0) or p.get("citation_count", 0) or 0,
                            open_access=bool(
                                p.get("open_access")
                                or p.get("openalex_is_oa")
                                or p.get("s2_open_access_pdf_url")
                            ),
                            full_text_url=(
                                p.get("openalex_best_oa_pdf_url")
                                or p.get("openalex_oa_url")
                                or p.get("s2_open_access_pdf_url")
                                or ""
                            ),
                        )

                    self._update_glance_corpus(len(corpus))
                    self._set_tab_badge(0, "Search", len(corpus))
                    if hasattr(self, '_populate_corpus_browser'):
                        self._populate_corpus_browser(corpus)
                    self._log(f"Loaded existing corpus: {len(corpus)} papers from {corpus_path}")
            except Exception as exc:
                self._log(f"Could not load corpus.json: {exc}")

        # 2. Load scored papers
        relevant_path = cfg.claims_dir / "relevant.json"
        if relevant_path.exists():
            try:
                with open(relevant_path, encoding="utf-8") as f:
                    relevant = _json.load(f)
                if isinstance(relevant, list):
                    n_relevant = len(relevant)
                    if hasattr(self, "_set_corpus_stage_relevance"):
                        self._set_corpus_stage_relevance("stage2", relevant)
                    self._update_glance_relevant(n_relevant)
                    self._set_tab_badge(1, "Score && Filter", n_relevant)
                    self._update_glance_stage("Score done")
                    self._log(f"Loaded existing scored papers: {n_relevant} from {relevant_path}")
            except Exception as exc:
                self._log(f"Could not load relevant.json: {exc}")

        self._load_existing_human_relevance_review(cfg)

        # 3. Load claims
        stats_path = cfg.claims_dir / "statistical_extractions.json"
        if stats_path.exists():
            try:
                with open(stats_path, encoding="utf-8") as f:
                    stats_artifact = _json.load(f)
                if isinstance(stats_artifact, dict):
                    summary = stats_artifact.get("summary", {})
                    self._log(
                        "Loaded statistical review: "
                        f"{summary.get('results', 0)} results, "
                        f"{summary.get('human_review_items', 0)} review items")
            except Exception as exc:
                self._log(f"Could not load statistical_extractions.json: {exc}")

        # 4. Load claims
        claims_path = cfg.claims_dir / "claims.json"
        if claims_path.exists():
            try:
                with open(claims_path, encoding="utf-8") as f:
                    claims = _json.load(f)
                if isinstance(claims, list):
                    self._log(f"Loaded existing claims: {len(claims)} from {claims_path}")
                    self._update_glance_stage("Extract done")
            except Exception as exc:
                self._log(f"Could not load claims.json: {exc}")

        # 5. Load filtered claims
        filtered_path = cfg.claims_dir / "claims_filtered.json"
        if filtered_path.exists():
            try:
                with open(filtered_path, encoding="utf-8") as f:
                    filtered = _json.load(f)
                if isinstance(filtered, list):
                    self._log(f"Loaded existing filtered claims: {len(filtered)} from {filtered_path}")
                    self._update_glance_stage("Audit done")
            except Exception as exc:
                pass

        # 6. Load robustness results
        robustness_path = cfg.data_dir / "validation" / "robustness_results.json"
        if robustness_path.exists():
            try:
                with open(robustness_path, encoding="utf-8") as f:
                    robustness = _json.load(f)
                if isinstance(robustness, dict) and robustness.get("n_papers"):
                    self._log(f"Loaded robustness results: {robustness.get('n_papers')} papers analysed")
                    self._update_glance_stage("Robustness done")
            except Exception as exc:
                pass

        self._refresh_pipeline_progress_from_disk(cfg)

    def _sync_fulltext_controls_to_config(self, *_args) -> None:
        """Persist visible full-text controls into the active Config object."""
        cfg = self.pipeline_config
        if hasattr(self, "sysys_ft_workers"):
            cfg.fulltext_download_workers = self.sysys_ft_workers.value()
        if hasattr(self, "sysys_ft_streaming"):
            cfg.fulltext_streaming = self.sysys_ft_streaming.isChecked()
        if hasattr(self, "sysys_ft_batch_size"):
            cfg.fulltext_download_batch_size = self.sysys_ft_batch_size.value()
        if hasattr(self, "sysys_ft_paperqa_enabled"):
            cfg.paperqa_index_enabled = self.sysys_ft_paperqa_enabled.isChecked()
        if hasattr(self, "sysys_ft_paperqa_batch_size"):
            cfg.paperqa_index_batch_size = self.sysys_ft_paperqa_batch_size.value()

    def _restore_fulltext_project_config(self, state: dict) -> bool:
        """Restore project-scoped full-text settings saved in project_state."""
        saved = state.get("config") or {}
        if not isinstance(saved, dict):
            return False
        restored = False
        bool_fields = ("fulltext_streaming", "paperqa_index_enabled")
        int_fields = (
            "fulltext_download_workers",
            "fulltext_download_batch_size",
            "paperqa_index_batch_size",
        )
        for field in bool_fields:
            value = saved.get(field)
            if isinstance(value, bool):
                setattr(self.pipeline_config, field, value)
                restored = True
        for field in int_fields:
            value = saved.get(field)
            if value is None:
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed >= 1:
                setattr(self.pipeline_config, field, parsed)
                restored = True
        return restored

    def _sync_settings_to_tabs(self, force: bool = False):
        """Push pipeline_config values into all per-stage tab widgets."""
        cfg = self.pipeline_config
        provider = cfg.llm_provider

        # Determine the model string for the active provider
        model_for_provider = {
            "ollama": cfg.ollama_model,
            "custom": cfg.custom_model,
            "openai": cfg.openai_model,
            "anthropic": cfg.anthropic_model,
        }
        model = model_for_provider.get(provider, "")

        # Helper to set a provider combo + model edit pair.
        # Normal project/open syncs preserve hand-edited stage rows. Preferences
        # Apply/Save passes force=True so the chosen provider/model becomes the
        # new baseline everywhere, including rows previously edited by hand.
        def _set_llm(combo: QComboBox, model_edit: QLineEdit):
            if force or not getattr(combo, "_user_customized", False):
                idx = combo.findText(provider)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                if force:
                    setattr(combo, "_user_customized", False)
            if force or (model and not getattr(model_edit, "_user_customized", False)):
                model_edit.setText(model)
                if force:
                    setattr(model_edit, "_user_customized", False)

        # Tab 0: Search — natural language question from research config
        nlq = cfg.get_natural_language_question()
        if nlq and not self.sysys_nl_query.text().strip():
            self.sysys_nl_query.setText(nlq)

        # Tab 0: Search — full-text preference from research config
        ft_cfg = cfg.get_full_text_config()
        if ft_cfg.get("keep_only_full_text"):
            self.fulltext_only.setChecked(True)
        if ft_cfg.get("enabled"):
            self.fetch_fulltext.setChecked(True)

        # Tab 1: Score & Filter
        relevance_model = (getattr(cfg, "relevance_model", "") or "").strip()
        if not force and relevance_model.lower().startswith(("brt", "housecatbert")):
            if not getattr(self.score_provider, "_user_customized", False):
                idx = self.score_provider.findText("brt-bert")
                if idx >= 0:
                    self.score_provider.setCurrentIndex(idx)
            if not getattr(self.score_model, "_user_customized", False):
                self.score_model.setText(relevance_model)
        else:
            _set_llm(self.score_provider, self.score_model)
        if hasattr(self, "score_relevance_model") and (
            force or not getattr(self.score_relevance_model, "_user_customized", False)
        ):
            self.score_relevance_model.setCurrentText(
                getattr(cfg, "relevance_model", "") or "brt-bert"
            )
            if force:
                setattr(self.score_relevance_model, "_user_customized", False)

        self.score_threshold.setValue(cfg.relevance_threshold)
        self.score_batch.setValue(cfg.score_batch_size)
        if hasattr(self, "stat_extract_provider"):
            _set_llm(self.stat_extract_provider, self.stat_extract_model)

        # Tab 2: Extract & Validate
        _set_llm(self.extract_provider, self.extract_model)
        self.extract_max_tokens.setValue(cfg.score_max_tokens)
        llm_sc = cfg.get_llm_scoring_config()
        if llm_sc.get("use_reasoning") is not None:
            self.extract_thinking.setChecked(bool(llm_sc["use_reasoning"]))
        _set_llm(self.validate_provider, self.validate_model)

        # Tab 3: Audit & Synthesize
        _set_llm(self.synth_provider, self.synth_model)
        syn_cfg = cfg.get_synthesis_config()
        if syn_cfg.get("n_clusters"):
            self.synth_min_cluster.setValue(syn_cfg["n_clusters"])

        # Tab 4: Fulltext & Interrater
        _set_llm(self.interrater_provider, self.interrater_model)
        ft_workers = cfg.fulltext_download_workers
        ft_streaming = bool(cfg.fulltext_streaming)
        ft_batch_size = cfg.fulltext_download_batch_size
        paperqa_enabled = bool(cfg.paperqa_index_enabled)
        paperqa_batch_size = cfg.paperqa_index_batch_size
        if hasattr(self, 'sysys_ft_workers'):
            self.sysys_ft_workers.setValue(ft_workers)
        if hasattr(self, "sysys_ft_streaming"):
            self.sysys_ft_streaming.setChecked(ft_streaming)
        if hasattr(self, "sysys_ft_batch_size"):
            self.sysys_ft_batch_size.setValue(ft_batch_size)
            self.sysys_ft_batch_size.setDisabled(ft_streaming)
        if hasattr(self, "sysys_ft_paperqa_enabled"):
            self.sysys_ft_paperqa_enabled.setChecked(paperqa_enabled)
        if hasattr(self, "sysys_ft_paperqa_batch_size"):
            self.sysys_ft_paperqa_batch_size.setValue(paperqa_batch_size)
            self.sysys_ft_paperqa_batch_size.setEnabled(paperqa_enabled)

        # Tab 5: Robustness & Verify
        _set_llm(self.verify_provider, self.verify_model)

        # All tabs: default paths from current project_root / data layout
        self._sync_all_stage_paths(cfg)

        # Stage 2: apply advanced-LLM visibility from QSettings
        self._apply_score_advanced_mode_from_settings()

        self._log("Settings synced to all stage tabs")

    def _apply_score_advanced_mode_from_settings(self):
        """Read QSettings and toggle Stage 2 advanced-LLM widgets accordingly."""
        from PySide6.QtCore import QSettings
        s = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        show_advanced = cast(bool, s.value("score/show_advanced_llm", False, type=bool))
        self._apply_score_advanced_mode(show_advanced)

    def _apply_score_advanced_mode(self, show_advanced: bool):
        """Show or hide Stage 2's provider/model controls.

        When hidden, Stage 2 uses the BRT-BERT relevance model unconditionally
        (enforced in `_start_sysys_score`). The simple-mode info label is shown
        in place of the controls.
        """
        if hasattr(self, "score_advanced_box"):
            self.score_advanced_box.setVisible(show_advanced)
        if hasattr(self, "score_simple_label"):
            self.score_simple_label.setVisible(not show_advanced)

    # ── Settings methods (delegate to SettingsDialog) ─────────────────
    #
    # These methods previously operated on widgets embedded in the
    # Settings tab.  They now operate on MainWindow.pipeline_config
    # directly, and the SettingsDialog handles the UI interaction.

    # ── Search and analysis worker launchers ──────────────────────────

    def _sync_analysis_articles_from_corpus(self, corpus: list) -> None:
        """Refresh keyword-analysis Article objects from the latest search corpus."""
        from core.models import Article

        self.articles = {}
        for idx, p in enumerate(corpus or []):
            if not isinstance(p, dict):
                continue
            art_id = p.get("id") or p.get("doi") or p.get("pmid") or p.get("title", "")[:60]
            if not art_id:
                art_id = f"article_{idx}"
            authors = p.get("authors", [])
            if not isinstance(authors, list):
                authors = []
            self.articles[str(art_id)] = Article(
                id=str(art_id),
                title=p.get("title", ""),
                authors=authors,
                abstract=p.get("abstract", "") or "",
                full_text=p.get("full_text", "") or "",
                year=p.get("year"),
                journal=p.get("journal", "") or "",
                doi=p.get("doi", "") or "",
                pmid=p.get("pmid", "") or "",
                citations=p.get("citationCount", 0) or p.get("citations", 0) or 0,
                open_access=bool(
                    p.get("open_access")
                    or p.get("is_oa")
                    or p.get("openalex_is_oa")
                    or p.get("s2_open_access_pdf_url")
                    or p.get("scopus_openaccess") == "1"
                ),
                full_text_url=(
                    p.get("openalex_best_oa_pdf_url")
                    or p.get("openalex_oa_url")
                    or p.get("s2_open_access_pdf_url")
                    or p.get("oa_url")
                    or p.get("full_text_url")
                    or ""
                ),
                has_full_text=bool(p.get("has_full_text") or p.get("full_text")),
                url=p.get("url", "") or "",
                keywords=p.get("keywords", []) if isinstance(p.get("keywords", []), list) else [],
            )

    def _start_fetch(self):
        """Fetch Articles button — download full text for the existing corpus.

        This is a download-only action: it does NOT run a new literature search.
        Use "Run Sysys Search" to build/extend the corpus first. To search and
        download in one step, tick the "Download full text" checkbox before
        running the search.
        """
        self._log("═" * 60)
        self._log("FETCH ARTICLES (DOWNLOAD FULL TEXT)")
        self._log("═" * 60)

        # Ensure a corpus is available; load from disk if the in-memory copy is
        # empty (e.g. right after opening a project).
        if not self._pipeline_corpus:
            corpus_path = self.pipeline_config.corpus_dir / "corpus.json"
            if corpus_path.exists():
                try:
                    import json as _json
                    with corpus_path.open(encoding="utf-8") as f:
                        self._pipeline_corpus = _json.load(f)
                    self._log(
                        f"Loaded {len(self._pipeline_corpus)} papers from {corpus_path}"
                    )
                except Exception as exc:
                    self._log(f"Could not load corpus: {exc}", "ERROR")
                    return

        if not self._pipeline_corpus:
            self._log(
                "No corpus to download. Run Sysys Search first to build a corpus.",
                "ERROR",
            )
            return

        # Delegate to corpus full-text enrichment (download only, no search).
        self._start_sysys_corpus_fulltext(triggered_by_fetch=True)

    def _start_analysis(self):
        """Run Keyword Analysis — score fetched articles against hypothesis components."""
        self._log("═" * 60)
        self._log("KEYWORD ANALYSIS")
        self._log("═" * 60)

        if not self.articles and not self._pipeline_corpus:
            self._log("No articles loaded. Run Sysys Search or Fetch Articles first.", "ERROR")
            return

        nl_query = self.sysys_nl_query.text().strip()
        if nl_query and not self.hypothesis_components:
            self._sync_query_components_from_question(nl_query)

        if not self.hypothesis_components:
            self._log(
                "No hypothesis components. Enter a question, edit components, "
                "or reset components first.",
                "ERROR",
            )
            return

        # Build Article objects from pipeline corpus if needed
        if not self.articles and self._pipeline_corpus:
            from core.models import Article
            for p in self._pipeline_corpus:
                art_id = p.get("doi") or p.get("pmid") or p.get("title", "")[:60]
                self.articles[art_id] = Article(
                    id=art_id,
                    title=p.get("title", ""),
                    authors=p.get("authors", []),
                    abstract=p.get("abstract", ""),
                    year=p.get("year"),
                    doi=p.get("doi", ""),
                    pmid=p.get("pmid", ""),
                    citations=p.get("citationCount", 0) or p.get("citations", 0) or 0,
                )

        self._log(f"Articles: {len(self.articles)}")
        self._log(f"Components: {len(self.hypothesis_components)}")
        self._log(f"Total pairs: {len(self.articles) * len(self.hypothesis_components)}")

        self.analysis_btn.setEnabled(False)
        self.search_progress.setVisible(True)
        self.search_progress.setRange(0, 0)

        # Run in a worker thread
        self._analysis_worker = AnalysisWorker(
            articles=self.articles,
            components=self.hypothesis_components,
        )
        self._analysis_worker.finished_analysis.connect(self._on_analysis_done)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.start()

    def _on_analysis_done(self, session):
        from analysis.scoring import update_component_weights_from_keyword_analysis

        self.session = session
        stats = update_component_weights_from_keyword_analysis(session)
        self.hypothesis_components = session.hypothesis_components
        self._refresh_component_list()
        self.analysis_btn.setEnabled(True)
        self.search_progress.setVisible(False)

        self._log("═" * 60)
        self._log(f"KEYWORD ANALYSIS COMPLETE")
        for ar in session.alignment_results:
            self._log(f"  {ar.component_label}: support={ar.support_score:.2f} "
                      f"({ar.supporting_count} sup, {ar.contradicting_count} contra, "
                      f"{ar.neutral_count} neutral)")
        if stats:
            ranked = sorted(
                self.hypothesis_components,
                key=lambda comp: getattr(comp, "weight", 1.0),
                reverse=True,
            )
            self._log("Updated query-term weights:")
            for comp in ranked[:12]:
                self._log(f"  {comp.label}: weight={comp.weight:.3f}")
        self._log("═" * 60)

        # Update dashboard if methods exist
        if hasattr(self, '_update_alignment_dashboard'):
            try:
                self._update_alignment_dashboard()
            except Exception:
                pass
        if hasattr(self, '_refresh_evidence_table'):
            try:
                self._refresh_evidence_table()
            except Exception:
                pass

    def _on_analysis_error(self, err: str):
        self._log(f"ANALYSIS ERROR: {err}", "ERROR")
        self.analysis_btn.setEnabled(True)
        self.search_progress.setVisible(False)

    def _start_fulltext_retrieval(self):
        """Run corpus full-text retrieval."""
        self._log("═" * 60)
        self._log("FULL-TEXT RETRIEVAL")
        self._log("═" * 60)
        if not self.articles:
            self._log("No articles loaded.", "ERROR")
            return
        self._log(f"Checking OA status for {len(self.articles)} articles...")
        self.ft_run_btn.setEnabled(False)
        self.ft_cancel_btn.setEnabled(True)
        self._ft_worker = FullTextRetrievalWorker(
            articles=self.articles,
            config=self.pipeline_config,
        )
        self._ft_worker.status.connect(lambda msg: self._log(msg))
        self._ft_worker.finished_retrieval.connect(self._on_ft_finished)
        self._ft_worker.error.connect(lambda e: self._log(f"FT retrieval error: {e}", "ERROR"))
        self._ft_worker.start()

    def _on_ft_finished(self, success_count: int, total: int, json_path: str):
        self.ft_run_btn.setEnabled(True)
        self.ft_cancel_btn.setEnabled(False)
        self._log(f"Full-text retrieval: {success_count}/{total} downloaded")
        if json_path:
            self._log(f"Output: {json_path}")

    def _start_text_scaling(self):
        """Run Wordfish text scaling + stylometric profiling."""
        self._log("═" * 60)
        self._log("TEXT SCALING & STYLOMETRY")
        self._log("═" * 60)
        if not self.articles:
            self._log("No articles loaded.", "ERROR")
            return
        min_words = self.wf_min_words.value() if hasattr(self, 'wf_min_words') else 50
        max_feat = self.wf_max_features.value() if hasattr(self, 'wf_max_features') else 5000
        self._log(f"Articles: {len(self.articles)}, min_words={min_words}, max_features={max_feat}")
        self._scaling_worker = TextScalingWorker(
            articles=self.articles,
            components=self.hypothesis_components,
            min_doc_words=min_words,
            max_features=max_feat,
        )
        self.wf_run_btn.setEnabled(False)
        self.wf_cancel_btn.setEnabled(True)
        self._scaling_worker.status.connect(lambda msg: self._log(msg))
        self._scaling_worker.finished_scaling.connect(self._on_scaling_finished)
        self._scaling_worker.cancelled.connect(self._on_scaling_cancelled)
        self._scaling_worker.error.connect(self._on_scaling_error)
        self._scaling_worker.start()

    # ── Search/full-text cancel methods ───────────────────────────────

    def _cancel_work(self):
        # "Fetch Articles" now runs the corpus full-text download via the
        # pipeline worker, so cancel that (and any legacy fetch worker).
        cancelled = False
        if self._pipeline_worker:
            self._pipeline_worker.cancel()
            cancelled = True
        if self._fetch_worker:
            # FetchWorker (legacy) has no cancel() — only QThread.terminate().
            self._fetch_worker.terminate()
            cancelled = True
        if cancelled:
            self._log("Fetch cancelled.")
        self.fetch_btn.setEnabled(True)
        self.cancel_fetch_btn.setEnabled(False)
        self.search_progress.setVisible(False)

    def _cancel_ft(self):
        if self._ft_worker:
            self._ft_worker.cancel()
        self.ft_run_btn.setEnabled(True)
        self.ft_cancel_btn.setEnabled(False)

    def _cancel_scaling(self):
        if self._scaling_worker:
            self._scaling_worker.cancel()
            self.wf_cancel_btn.setEnabled(False)
            self._log("Cancelling text scaling...")

    # ── Search/full-text signal handler slots ─────────────────────────

    def _on_error(self, msg):
        self._log(msg, level="ERROR")
        QMessageBox.warning(self, "Error", msg)

    def _on_scaling_finished(self, model, profiles: dict, stats: dict):
        self._wordfish_model = model
        self._stylometry_profiles = profiles
        self._corpus_stats = stats
        self.session.wordfish_model = model
        self.session.stylometry_profiles = profiles
        self.session.corpus_statistics = stats
        self.wf_run_btn.setEnabled(True)
        self.wf_cancel_btn.setEnabled(False)
        self._populate_wf_table(model)
        self._populate_sty_table(profiles)
        self._update_sty_kpis(stats)
        self._log(
            f"Text scaling complete: {len(profiles)} profiles, "
            f"{stats.get('n_features', 0)} features"
        )

    def _on_scaling_cancelled(self):
        self.wf_run_btn.setEnabled(True)
        self.wf_cancel_btn.setEnabled(False)
        self._log("Text scaling cancelled.", "WARNING")

    def _on_scaling_error(self, message: str):
        self.wf_run_btn.setEnabled(True)
        self.wf_cancel_btn.setEnabled(False)
        self._log(f"Scaling error: {message}", "ERROR")

    # ── Search/full-text populate / refresh / update display methods ──

    def _populate_wf_table(self, model):
        positions = model.get("positions", {}) if isinstance(model, dict) else {}
        self.wf_table.setSortingEnabled(False)
        self.wf_table.setRowCount(len(positions))
        for row, (article_id, position) in enumerate(
            sorted(positions.items(), key=lambda item: item[1])
        ):
            article = self.articles.get(article_id)
            if isinstance(article, dict):
                title = article.get("title", article_id)
            else:
                title = getattr(article, "title", article_id)
            title_item = QTableWidgetItem(str(title or article_id))
            title_item.setData(Qt.ItemDataRole.UserRole, article_id)
            self.wf_table.setItem(row, 0, title_item)
            self.wf_table.setItem(row, 1, QTableWidgetItem(f"{float(position):+.4f}"))
            if position < -0.1:
                direction = "Negative pole"
            elif position > 0.1:
                direction = "Positive pole"
            else:
                direction = "Center"
            self.wf_table.setItem(row, 2, QTableWidgetItem(direction))
        self.wf_table.setSortingEnabled(True)
        self.wf_docs_card.set_value(str(model.get("n_documents", len(positions))))
        self.wf_features_card.set_value(str(model.get("n_features", 0)))
        convergence = "Yes" if model.get("converged") else "No"
        self.wf_convergence_card.set_value(
            f"{convergence} ({model.get('iterations', 0)})"
        )
        terms = model.get("discriminating_terms", [])
        negative_terms = sorted(
            (term for term in terms if term.get("beta", 0.0) < 0.0),
            key=lambda term: term.get("beta", 0.0),
        )[:10]
        positive_terms = sorted(
            (term for term in terms if term.get("beta", 0.0) > 0.0),
            key=lambda term: term.get("beta", 0.0),
            reverse=True,
        )[:10]
        negative_text = ", ".join(
            f"{term.get('term')} ({term.get('beta', 0.0):+.3f})"
            for term in negative_terms
        ) or "None"
        positive_text = ", ".join(
            f"{term.get('term')} ({term.get('beta', 0.0):+.3f})"
            for term in positive_terms
        ) or "None"
        self.wf_terms_detail.setPlainText(
            f"Negative pole: {negative_text}\nPositive pole: {positive_text}"
        )

    def _populate_sty_table(self, profiles: dict):
        ordered_profiles = sorted(
            profiles.items(),
            key=lambda item: str(item[1].get("title", item[0])).lower(),
        )
        self.sty_table.setSortingEnabled(False)
        self.sty_table.setRowCount(len(ordered_profiles))
        for row, (article_id, profile) in enumerate(ordered_profiles):
            title_item = QTableWidgetItem(str(profile.get("title") or article_id))
            title_item.setData(Qt.ItemDataRole.UserRole, article_id)
            self.sty_table.setItem(row, 0, title_item)
            self.sty_table.setItem(
                row, 1, QTableWidgetItem(f"{profile.get('hedge_density', 0.0):.2f}")
            )
            self.sty_table.setItem(
                row, 2, QTableWidgetItem(f"{profile.get('certainty_density', 0.0):.2f}")
            )
            self.sty_table.setItem(
                row, 3, QTableWidgetItem(f"{profile.get('flesch_kincaid_grade', 0.0):.1f}")
            )
            self.sty_table.setItem(
                row, 4, QTableWidgetItem(f"{profile.get('type_token_ratio', 0.0):.3f}")
            )
        self.sty_table.setSortingEnabled(True)

    def _update_sty_kpis(self, stats: dict):
        self.sty_corpus_card.set_value(str(stats.get("n_documents", 0)))
        self.sty_avg_hedge_card.set_value(
            f"{stats.get('average_hedge_density', 0.0):.2f}"
        )
        self.sty_avg_certainty_card.set_value(
            f"{stats.get('average_certainty_density', 0.0):.2f}"
        )

    def _on_sty_row_clicked(self, row: int, col: int):
        title_item = self.sty_table.item(row, 0)
        if title_item is None:
            return
        article_id = title_item.data(Qt.ItemDataRole.UserRole)
        profile = self._stylometry_profiles.get(str(article_id), {})
        if not profile:
            return
        self.sty_detail.setPlainText(
            f"{profile.get('title', article_id)}\n"
            f"Words: {profile.get('word_count', 0):,} | "
            f"Sentences: {profile.get('sentence_count', 0):,} | "
            f"Average sentence length: {profile.get('average_sentence_length', 0.0):.1f}\n"
            f"Hedges per 100 words: {profile.get('hedge_density', 0.0):.2f} | "
            f"Certainty terms per 100 words: {profile.get('certainty_density', 0.0):.2f} | "
            f"Passive constructions per 100 sentences: "
            f"{profile.get('passive_voice_density', 0.0):.2f}"
        )

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child = item.layout()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif child is not None:
                self._clear_layout(child)

    def _populate_alignment_bars(self, results) -> None:
        """Fill the per-component AlignmentBar rows from alignment results."""
        self._clear_layout(self.alignment_bars_layout)
        self._grade_badges = []
        if not results:
            empty = QLabel(
                "No per-component alignment yet. Run Keyword Analysis first."
            )
            empty.setWordWrap(True)
            self.alignment_bars_layout.addWidget(empty)
            return
        for result in results:
            row = QHBoxLayout()
            name = QLabel(result.component_label or result.component_id)
            name.setMinimumWidth(180)
            name.setWordWrap(True)
            bar = AlignmentBar()
            bar._component_id = result.component_id
            score = float(result.support_score or 0.0)
            bar.set_score(score, f"{score:+.2f}")
            row.addWidget(name)
            row.addWidget(bar, stretch=1)
            self.alignment_bars_layout.addLayout(row)

    def _update_alignment_dashboard(self):
        session = getattr(self, "session", None)
        results = list(getattr(session, "alignment_results", None) or [])
        if not results and getattr(self, "_grade_results", None):
            from core.models import AlignmentResult
            results = [
                AlignmentResult(
                    component_id=grade.component_id,
                    component_label=grade.component_label,
                )
                for grade in self._grade_results
            ]
        self._populate_alignment_bars(results)
        articles = getattr(session, "articles", None) or {}
        analyzed = int(getattr(session, "total_articles_analyzed", 0) or len(articles))
        supporting = sum(int(getattr(item, "supporting_count", 0) or 0) for item in results)
        contradicting = sum(int(getattr(item, "contradicting_count", 0) or 0) for item in results)
        overall = (
            sum(float(getattr(item, "support_score", 0.0) or 0.0) for item in results) / len(results)
            if results else 0.0
        )
        fetched = int(getattr(session, "total_articles_fetched", 0) or 0)
        self.dash_total_card.set_value(str(len(articles) or fetched))
        self.dash_analyzed_card.set_value(str(analyzed))
        self.dash_supporting_card.set_value(str(supporting))
        self.dash_contradicting_card.set_value(str(contradicting))
        self.dash_overall_card.set_value(f"{overall:+.2f}")
        if getattr(self, "_grade_results", None):
            self._attach_grade_badges()
        self._log("Updating alignment dashboard...")

    def _refresh_evidence_table(self):
        from process.evidence_table import load_evidence_rows

        self.ev_refresh_btn.setEnabled(False)
        self.ev_status_label.setText("Loading evidence...")
        QApplication.processEvents()
        try:
            rows, source = load_evidence_rows(
                self.pipeline_config,
                session=self.session,
                articles=self.articles,
                components=self.hypothesis_components,
            )
            self._evidence_table_rows = rows
            self._evidence_table_source = source

            current_component = self.ev_component_filter.currentText()
            components = sorted({
                str(row.get("component") or "Primary question")
                for row in rows
            })
            self.ev_component_filter.blockSignals(True)
            self.ev_component_filter.clear()
            self.ev_component_filter.addItem("All")
            self.ev_component_filter.addItems(components)
            if current_component in components:
                self.ev_component_filter.setCurrentText(current_component)
            self.ev_component_filter.blockSignals(False)

            self._apply_evidence_filters()
            if rows:
                self.ev_refresh_btn.setText("Refresh Evidence Table")
                self._log(f"Evidence table loaded: {len(rows)} rows from {source}")
            else:
                self.ev_status_label.setText(
                    "No evidence found. Run Keyword Analysis or Stage 3 Extract Claims first."
                )
                self._log("No evidence available for the Evidence Table.", "WARNING")
        except Exception as exc:
            self._evidence_table_rows = []
            self._evidence_table_source = ""
            self.evidence_table.setRowCount(0)
            self.ev_status_label.setText(f"Could not load evidence: {exc}")
            self._log(f"Evidence table load failed: {exc}", "ERROR")
        finally:
            self.ev_refresh_btn.setEnabled(True)

    def _apply_evidence_filters(self, *_):
        direction_filter = self.ev_direction_filter.currentText().lower()
        component_filter = self.ev_component_filter.currentText()
        filtered_rows = [
            row for row in self._evidence_table_rows
            if (direction_filter == "all" or row.get("direction") == direction_filter)
            and (component_filter == "All" or row.get("component") == component_filter)
        ]

        self.evidence_table.setSortingEnabled(False)
        self.evidence_table.setRowCount(len(filtered_rows))
        for table_row, evidence in enumerate(filtered_rows):
            title_item = QTableWidgetItem(str(evidence.get("title") or "Untitled"))
            title_item.setData(Qt.ItemDataRole.UserRole, evidence)
            self.evidence_table.setItem(table_row, 0, title_item)
            self.evidence_table.setItem(
                table_row,
                1,
                QTableWidgetItem(str(evidence.get("component") or "Primary question")),
            )
            score_item = QTableWidgetItem()
            score_item.setData(
                Qt.ItemDataRole.DisplayRole,
                round(float(evidence.get("score") or 0.0), 3),
            )
            self.evidence_table.setItem(table_row, 2, score_item)

            direction = str(evidence.get("direction") or "neutral")
            direction_item = QTableWidgetItem(direction)
            if direction == "supports":
                direction_item.setForeground(QColor(COLORS["success"]))
            elif direction == "contradicts":
                direction_item.setForeground(QColor(COLORS["error"]))
            self.evidence_table.setItem(table_row, 3, direction_item)

            excerpt = str(evidence.get("excerpt") or "")
            self.evidence_table.setItem(
                table_row,
                4,
                QTableWidgetItem(excerpt[:260] + ("..." if len(excerpt) > 260 else "")),
            )
        self.evidence_table.setSortingEnabled(True)

        source_text = self._evidence_table_source or "no source"
        self.ev_status_label.setText(
            f"Showing {len(filtered_rows):,} of {len(self._evidence_table_rows):,} "
            f"evidence rows from {source_text}"
        )

    def _on_evidence_cell_clicked(self, row, col):
        title_item = self.evidence_table.item(row, 0)
        evidence = title_item.data(Qt.ItemDataRole.UserRole) if title_item else None
        if not isinstance(evidence, dict):
            return

        details = [
            str(evidence.get("title") or "Untitled"),
            "",
            f"Component: {evidence.get('component') or 'Primary question'}",
            f"Direction: {evidence.get('direction') or 'neutral'}",
            f"Score: {float(evidence.get('score') or 0.0):.3f}",
        ]
        citation_bits = []
        if evidence.get("year"):
            citation_bits.append(str(evidence["year"]))
        if evidence.get("doi"):
            citation_bits.append(f"DOI {evidence['doi']}")
        if evidence.get("pmid"):
            citation_bits.append(f"PMID {evidence['pmid']}")
        if citation_bits:
            details.append("Citation: " + " | ".join(citation_bits))
        if evidence.get("authors"):
            details.append(f"Authors: {evidence['authors']}")

        for label, key in (
            ("Mechanisms", "mechanisms"),
            ("Conditions", "conditions"),
            ("Key entities", "key_entities"),
            ("Matched keywords", "matched_keywords"),
        ):
            values = evidence.get(key) or []
            if values:
                details.append(f"{label}: {', '.join(str(value) for value in values)}")
        if evidence.get("excerpt"):
            details.extend(["", "Evidence excerpt:", str(evidence["excerpt"])])
        self.evidence_detail.setPlainText("\n".join(details))

    # ── Sysys search launcher and handlers ───────────────────────────

    def _on_search_question_changed(self, text: str) -> None:
        self._update_search_workflow_buttons()
        self._original_question_state.set_prospective(text)

    def _capture_original_question_for_first_search(
        self, query: str, *, save: bool = True
    ) -> str:
        original = self._original_question_state.capture_first_search(query)
        if save and self._project_path is not None:
            self._save_project()
        return original

    def _original_question_for_model(self) -> str:
        current = self.sysys_nl_query.text()
        original = self._original_question_state.question_for_model(current)
        # Model-backed stages can run before Stage 1 (e.g. against a loaded
        # corpus). Freeze the fallback question on first non-empty use so
        # later stages cannot silently drift to a different GUI value.
        if original and not self._original_question_state.captured:
            self._capture_original_question_for_first_search(original)
        return original

    def _edit_original_research_question(self, question: str) -> str:
        original = self._original_question_state.edit_captured(question)
        if self._project_path is not None:
            self._save_project()
        return original

    def _project_question_window_state(self) -> dict[str, object]:
        return {
            "search_query": self.sysys_nl_query.text(),
            **self._original_question_state.to_window_state(),
        }

    def _start_sysys_search(self):
        if not self._pipeline_stage_guard("Stage 1 (Search)"):
            return
        self._log("═" * 60)
        self._log("STAGE 1: LITERATURE SEARCH")
        self._log("═" * 60)

        cfg = self.pipeline_config
        nl_query = self.sysys_nl_query.text().strip()

        if not nl_query:
            self._log("No search query provided.", "WARNING")
            return
        # Capture the original before syncing (model-backed sync may consult
        # it), but defer the project save until after sync so the persisted
        # state reflects the just-computed hypothesis components rather than
        # the stale pre-sync list.
        self._capture_original_question_for_first_search(nl_query, save=False)
        self._sync_query_components_from_question(nl_query)
        if self._project_path is not None:
            self._save_project()

        # Gather which providers are selected (from checkboxes)
        selected_providers = []
        if self.src_pubmed.isChecked():
            selected_providers.append("PubMed")
        if self.src_s2.isChecked():
            selected_providers.append("Semantic Scholar")
        if self.src_openalex.isChecked():
            selected_providers.append("OpenAlex")
        if self.src_elicit.isChecked():
            selected_providers.append("Elicit")
        if self.src_scopus.isChecked():
            selected_providers.append("Scopus")
        if self.src_wos.isChecked():
            selected_providers.append("Web of Science")

        paper_limit = self.paper_limit.value()
        self._log(f"Query: {nl_query}")
        self._log(f"Providers: {', '.join(selected_providers) or 'NONE'}")
        self._log(f"Paper limit: {paper_limit} total across selected providers")
        self._log(f"Config project root: {cfg.project_root}")
        self._log(f"Config corpus dir: {cfg.corpus_dir}")
        self._log(f"LLM provider: {cfg.llm_provider} / "
                   f"{getattr(cfg, 'llm_model', 'N/A')}")
        self._log(f"API keys configured: "
                   f"PubMed={'yes' if cfg.pubmed_api_key else 'NO'}, "
                   f"PubMed email={'yes' if cfg.pubmed_email else 'NO'}, "
                   f"S2={'yes' if cfg.s2_api_key else 'NO'}, "
                   f"OpenAI={'yes' if cfg.openai_api_key else 'NO'}, "
                   f"Custom={'yes' if cfg.custom_api_key else 'NO'}, "
                   f"Anthropic={'yes' if cfg.anthropic_api_key else 'NO'}, "
                   f"Elicit={'yes' if cfg.elicit_api_key else 'NO'}, "
                   f"Scopus={'yes' if cfg.scopus_api_key else 'NO'}, "
                   f"WoS={'yes' if cfg.wos_api_key else 'NO'}")

        if not selected_providers:
            self._log("No search providers selected!", "WARNING")
            return

        # Determine provider enum(s) for search_runner
        from gui.search_runner import ProviderChoice

        provider_map = {
            "PubMed": ProviderChoice.PUBMED,
            "Semantic Scholar": ProviderChoice.S2,
            "OpenAlex": ProviderChoice.OPENALEX,
            "Elicit": ProviderChoice.ELICIT,
            "Scopus": ProviderChoice.SCOPUS,
            "Web of Science": ProviderChoice.WOS,
        }
        providers = [provider_map[name] for name in selected_providers
                     if name in provider_map]
        if not providers:
            providers = [ProviderChoice.ALL]

        self._log(f"Provider enums: {[p.value for p in providers]}")
        self._log("Creating PipelineSearchWorker...")
        from analysis.scoring import (
            build_weighted_boolean_query,
            build_weighted_component_query_terms,
        )

        weighted_terms = build_weighted_component_query_terms(self.hypothesis_components)
        boolean_query = build_weighted_boolean_query(self.hypothesis_components)
        semantic_terms = weighted_terms[:10] or [nl_query]
        boolean_terms = [boolean_query] if boolean_query else []
        if boolean_query and hasattr(self, "boolean_query_preview"):
            self.boolean_query_preview.setPlainText(boolean_query)
        if boolean_query:
            self._log(f"Weighted Boolean query: {boolean_query}")
        self._search_worker = PipelineSearchWorker(
            natural_language_question=nl_query,
            semantic_terms=semantic_terms,
            boolean_terms=boolean_terms,
            provider=providers,
            config=cfg,
            max_results=paper_limit,
        )
        self._search_worker.progress.connect(self._on_sysys_search_progress)
        self._search_worker.finished_search.connect(self._on_sysys_search_done)
        self._search_worker.error.connect(self._on_sysys_search_error)

        self.sysys_search_btn.setEnabled(False)
        self.sysys_cancel_btn.setEnabled(True)
        self.pipeline_state = PipelineState.SEARCHING
        self.search_progress.setVisible(True)
        self.search_progress.setRange(0, 0)  # indeterminate

        self._log("Starting search worker thread...")
        self._set_pipeline_stage(1, "running")
        self._search_worker.start()

    def _cancel_sysys_search(self):
        self._log("Cancel requested for Sysys search/full-text work")
        if self._search_worker:
            self._search_worker.cancel()
            self._log("Sent cancel signal to search worker")
        if self._pipeline_worker:
            self._pipeline_worker.cancel()
            self._log("Sent cancel signal to pipeline worker")
        self.sysys_search_btn.setEnabled(True)
        self.sysys_cancel_btn.setEnabled(False)
        self.search_progress.setVisible(False)
        self.pipeline_state = PipelineState.IDLE

    def _on_sysys_search_progress(self, msg: str):
        self.search_status.setText(msg)
        self._log(msg)

    def _on_sysys_search_counts(self, current: int, total: int):
        total = max(1, total)
        current = min(max(0, current), total)
        self.search_progress.setRange(0, total)
        self.search_progress.setValue(current)

    def _on_sysys_search_done(self, corpus: list):
        self._pipeline_corpus = corpus
        self._sync_analysis_articles_from_corpus(corpus)
        self._sysys_search_runs += 1
        self.sysys_search_btn.setEnabled(True)
        self.sysys_cancel_btn.setEnabled(False)
        self.search_progress.setVisible(False)
        self.pipeline_state = PipelineState.IDLE

        # Feature D: update At-a-Glance corpus count
        self._update_glance_corpus(len(corpus))
        self._update_glance_stage("Search done")

        # Feature F: tab badge
        self._set_tab_badge(0, "Search", len(corpus))

        # Feature E: inline results preview (first 10 articles)
        lines = [f"Search complete — {len(corpus)} unique papers\n"]
        for i, p in enumerate(corpus[:10], 1):
            title = p.get("title", "(no title)")[:80]
            year = p.get("year", "")
            source = p.get("source", "")
            lines.append(f"{i:2}. [{year}] {title}  ({source})")
        if len(corpus) > 10:
            lines.append(f"… and {len(corpus) - 10} more")
        self.search_results_display.setPlainText("\n".join(lines))

        n_papers = len(corpus)
        label = "paper" if n_papers == 1 else "papers"
        self._set_pipeline_stage(1, "done", summary=f"{n_papers} {label}")

        # Feature B: populate corpus browser
        self._populate_corpus_browser(corpus)

        self._log("═" * 60)
        self._log(f"SEARCH COMPLETE: {len(corpus)} unique papers")
        self._log("═" * 60)

        # Source breakdown
        sources: dict[str, int] = {}
        for p in corpus:
            src = p.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        for src, count in sorted(sources.items()):
            self._log(f"  {src}: {count} papers")

        # Year distribution
        years = [p.get("year", 0) for p in corpus if p.get("year")]
        if years:
            self._log(f"  Year range: {min(years)} - {max(years)}")

        # Papers with/without abstracts
        with_abstract = sum(1 for p in corpus if p.get("abstract"))
        self._log(f"  With abstracts: {with_abstract}/{len(corpus)}")

        # Save path
        save_path = self.pipeline_config.corpus_dir / "corpus.json"
        if save_path.exists():
            size_kb = save_path.stat().st_size / 1024
            self._log(f"  Saved to: {save_path} ({size_kb:.1f} KB)")
        else:
            self._log(f"  Expected save path: {save_path}")

        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 1 search complete")
        self._keyword_analysis_ready = bool(corpus or self.articles)
        self._update_search_workflow_buttons()
        if (
            self.fetch_fulltext.isChecked()
            and corpus
            and not self._run_all_active
        ):
            self._start_sysys_corpus_fulltext(auto_started=True)

        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_sysys_search_error(self, err: str):
        self._log(f"SEARCH ERROR: {err}", "ERROR")
        self._on_error(err)
        self._set_pipeline_stage(1, "failed", summary=self._pipeline_error_summary(err))
        self.sysys_search_btn.setEnabled(True)
        self.sysys_cancel_btn.setEnabled(False)
        self.search_progress.setVisible(False)
        self.pipeline_state = PipelineState.IDLE
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 1 (search error).", "ERROR")

    # ── Generic pipeline stage infrastructure ─────────────────────────

    def _custom_endpoint(self) -> CustomEndpoint:
        """The custom endpoint from Settings, used by every stage's Fetch button."""
        cfg = self.pipeline_config
        return CustomEndpoint(cfg.custom_api_format, cfg.custom_base_url, cfg.custom_api_key)

    def _apply_config_overrides_for_stage(
        self, provider_combo: QComboBox, model_edit: QLineEdit
    ) -> Config:
        import copy
        cfg = copy.deepcopy(self.pipeline_config)
        original = MainWindow._original_question_for_model(self)
        if original:
            # Route the original question through `_research_config` /
            # `get_natural_language_question()` only. Setting
            # `cfg.hypothesis_text` here would leak it into
            # `build_hypothesis_components` (via `_sync_query_components_
            # from_question`) and silently change search behavior whenever
            # LLM Context Check is enabled.
            research_config = dict(getattr(cfg, "_research_config", {}) or {})
            research_config["natural_language_question"] = original
            cfg._research_config = research_config
        prov = provider_combo.currentText()
        model = model_edit.text().strip()
        if prov in ("brt-bert", "brt-irr", "housecatbert-scibert-releevance"):
            cfg.relevance_model = (
                model
                if model.lower().startswith(("brt", "housecatbert"))
                else prov
            )
            return cfg
        prov = normalize_provider_name(prov)
        cfg.llm_provider = prov
        if model:
            setattr(cfg, PROVIDER_MODEL_FIELD.get(prov, "openai_model"), model)
        return cfg

    def _cancel_pipeline_stage(self):
        if self._pipeline_worker:
            self._pipeline_worker.cancel()
        if self._search_worker:
            self._search_worker.cancel()
        # Immediately re-enable buttons so the user isn't stuck
        run_btn = getattr(self, '_active_stage_run_btn', None)
        cancel_btn = getattr(self, '_active_stage_cancel_btn', None)
        progress_bar = getattr(self, '_active_stage_progress', None)
        if run_btn:
            run_btn.setEnabled(True)
        if cancel_btn:
            cancel_btn.setEnabled(False)
        if progress_bar:
            progress_bar.setVisible(False)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL cancelled by user.")
        self._log("Stage cancelled")

    # ── Sysys individual stage launchers ─────────────────────────────

    def _score_corpus_path(self) -> Path:
        raw = self.score_corpus_path.text().strip()
        return Path(raw) if raw else self.pipeline_config.corpus_dir / "corpus.json"

    def _human_relevance_artifact_path(self) -> Path:
        return self.pipeline_config.data_dir / "validation" / "human_relevance_review.json"

    def _human_relevance_summary_path(self) -> Path:
        return self.pipeline_config.data_dir / "validation" / "human_relevance_agreement.json"

    def _load_json_list(self, path: Path) -> list[dict]:
        import json as _json

        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {path}")
        return data

    def _load_existing_human_relevance_review(self, cfg) -> None:
        """Restore the current project's saved review without clobbering edits."""
        import json as _json

        review_path = cfg.data_dir / "validation" / "human_relevance_review.json"
        try:
            resolved_path = review_path.resolve()
        except OSError:
            resolved_path = review_path
        if self._loaded_human_review_path == resolved_path:
            return

        self._loaded_human_review_path = resolved_path
        if not review_path.exists():
            self._human_review_artifact = None
            self._populate_human_relevance_table([])
            self.human_review_status.setText("No human queue loaded")
            return

        try:
            with open(review_path, encoding="utf-8") as f:
                artifact = _json.load(f)
            if not isinstance(artifact, dict):
                raise ValueError("Expected a review artifact object")
            rows = artifact.get("records", [])
            if not isinstance(rows, list) or not all(
                isinstance(row, dict) for row in rows
            ):
                raise ValueError("Expected review records to be a list of objects")
            self._human_review_artifact = artifact
            self._populate_human_relevance_table(rows)
            reviewed = artifact.get(
                "reviewed_count",
                sum(1 for row in rows if row.get("human_score") is not None),
            )
            verified = artifact.get(
                "verified_count",
                sum(1 for row in rows if row.get("human_verified")),
            )
            self.human_review_status.setText(
                f"Loaded {reviewed}/{len(rows)} human scores; "
                f"{verified} human verified"
            )
            self._log(f"Loaded human relevance review: {review_path}")
        except Exception as exc:
            self._human_review_artifact = None
            self._populate_human_relevance_table([])
            self.human_review_status.setText("Could not load saved human review")
            self._log(f"Could not load human relevance review: {exc}", "ERROR")

    def _set_human_relevance_editing(self, enabled: bool) -> None:
        """Toggle direct editing for the correctable review-table cells."""
        triggers = (
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.SelectedClicked
            | QTableWidget.EditTrigger.EditKeyPressed
        )
        self.human_review_table.setEditTriggers(
            triggers if enabled else QTableWidget.EditTrigger.NoEditTriggers
        )
        for row_index in range(self.human_review_table.rowCount()):
            for column in range(2, 7):
                item = self.human_review_table.item(row_index, column)
                if item is None:
                    continue
                flags = item.flags()
                if enabled:
                    flags |= Qt.ItemFlag.ItemIsEditable
                else:
                    flags &= ~Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)

    def _populate_human_relevance_table(self, rows: list[dict]) -> None:
        self._human_review_rows = rows
        self.human_review_table.setSortingEnabled(False)
        self.human_review_table.setRowCount(0)
        for row_index, row in enumerate(rows):
            self.human_review_table.insertRow(row_index)

            score_combo = QComboBox()
            score_combo.addItems(["", "0", "1", "2", "3", "4", "5"])
            human_score = row.get("human_score")
            if isinstance(human_score, int) and 0 <= human_score <= 5:
                score_combo.setCurrentText(str(human_score))
            self.human_review_table.setCellWidget(row_index, 0, score_combo)

            verified_item = QTableWidgetItem()
            verified_item.setFlags(
                (verified_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                & ~Qt.ItemFlag.ItemIsEditable
            )
            verified_item.setCheckState(
                Qt.CheckState.Checked
                if bool(row.get("human_verified"))
                else Qt.CheckState.Unchecked
            )
            verified_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            verified_item.setToolTip("Mark this row as checked by a human reviewer")
            verified_item.setData(Qt.ItemDataRole.UserRole, row_index)
            self.human_review_table.setItem(row_index, 1, verified_item)

            values = [
                row.get("title", ""),
                row.get("year", ""),
                row.get("model_score", ""),
                row.get("article_id", ""),
                row.get("abstract", ""),
            ]
            for offset, value in enumerate(values, start=2):
                item = QTableWidgetItem(str(value if value is not None else ""))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setData(Qt.ItemDataRole.UserRole, row_index)
                self.human_review_table.setItem(row_index, offset, item)
        self._set_human_relevance_editing(self.human_review_edit_cells.isChecked())
        self.human_review_table.setSortingEnabled(False)
        self.human_review_table.resizeColumnToContents(0)
        self.human_review_table.resizeColumnToContents(1)
        self.human_review_table.resizeColumnToContents(3)
        self.human_review_table.resizeColumnToContents(4)
        self.human_review_table.resizeColumnToContents(5)

    def _collect_human_relevance_rows(self) -> list[dict]:
        def _text(row_index: int, column: int) -> str:
            item = self.human_review_table.item(row_index, column)
            return item.text().strip() if item is not None else ""

        def _number_or_text(value: str, *, integer_only: bool = False):
            if not value:
                return None
            try:
                return int(value) if integer_only else float(value)
            except ValueError:
                return value

        rows: list[dict] = []
        for row_index, row in enumerate(self._human_review_rows):
            updated = dict(row)
            widget = self.human_review_table.cellWidget(row_index, 0)
            if isinstance(widget, QComboBox):
                value = widget.currentText().strip()
                updated["human_score"] = int(value) if value else None
            verified_item = self.human_review_table.item(row_index, 1)
            updated["human_verified"] = bool(
                verified_item is not None
                and verified_item.checkState() == Qt.CheckState.Checked
            )
            updated["title"] = _text(row_index, 2)
            updated["year"] = _number_or_text(
                _text(row_index, 3), integer_only=True
            )
            model_score = _number_or_text(_text(row_index, 4))
            if isinstance(model_score, float) and model_score.is_integer():
                model_score = int(model_score)
            if model_score != row.get("model_score"):
                updated["model_score_human_corrected"] = True
            updated["model_score"] = model_score
            updated["article_id"] = _text(row_index, 5)
            updated["abstract"] = _text(row_index, 6)
            rows.append(updated)
        self._human_review_rows = rows
        return rows

    def _build_human_relevance_queue(self) -> None:
        from process.human_relevance_review import build_review_queue, recommended_sample_size

        corpus_path = self._score_corpus_path()
        if not corpus_path.exists():
            self._log("No corpus found for human calibration queue. Run search first.", "ERROR")
            return
        try:
            papers = self._load_json_list(corpus_path)
            requested = self.human_review_sample.value()
            sample_size = requested or recommended_sample_size(len(papers))
            artifact = build_review_queue(
                papers,
                sample_size=sample_size,
                seed=42,
                source_path=str(corpus_path),
            )
            self._human_review_artifact = artifact
            try:
                self._loaded_human_review_path = (
                    self._human_relevance_artifact_path().resolve()
                )
            except OSError:
                self._loaded_human_review_path = self._human_relevance_artifact_path()
            self._populate_human_relevance_table(artifact.get("records", []))
            self.human_review_status.setText(
                f"Queue ready: {artifact.get('sample_size', 0)} of "
                f"{artifact.get('corpus_size', 0)} papers"
            )
            self._log(
                "Human relevance queue ready: "
                f"{artifact.get('sample_size', 0)} papers from {corpus_path}"
            )
        except Exception as exc:
            self._log(f"Could not build human relevance queue: {exc}", "ERROR")

    def _save_human_relevance_review(self) -> None:
        from core.io import atomic_write_json

        if not self._human_review_artifact:
            self._log("No human relevance queue to save. Build a queue first.", "ERROR")
            return
        rows = self._collect_human_relevance_rows()
        artifact = dict(self._human_review_artifact)
        artifact["records"] = rows
        artifact["reviewed_count"] = sum(1 for row in rows if row.get("human_score") is not None)
        artifact["verified_count"] = sum(
            1 for row in rows if row.get("human_verified")
        )
        out_path = self._human_relevance_artifact_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_path, artifact)
        self._human_review_artifact = artifact
        try:
            self._loaded_human_review_path = out_path.resolve()
        except OSError:
            self._loaded_human_review_path = out_path
        self.human_review_status.setText(
            f"Saved {artifact['reviewed_count']}/{len(rows)} human scores; "
            f"{artifact['verified_count']} human verified"
        )
        self._log(f"Human relevance review saved: {out_path}")

    def _scored_relevance_path(self) -> Path | None:
        candidates = [
            self.pipeline_config.claims_dir / "scored_all.json",
            self.pipeline_config.claims_dir / "relevant.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _refresh_human_relevance_model_scores(self) -> None:
        if not self._human_review_rows:
            return
        scored_path = self._scored_relevance_path()
        if scored_path is None:
            return
        try:
            from process.human_relevance_review import article_id

            scored = self._load_json_list(scored_path)
            by_id: dict[str, dict] = {}
            for index, paper in enumerate(scored):
                key = article_id(paper, index)
                by_id[key] = paper
                title = str(paper.get("title") or "").strip()
                if title:
                    by_id.setdefault(title[:120], paper)
            updated = []
            for row in self._collect_human_relevance_rows():
                match = by_id.get(str(row.get("article_id") or ""))
                if match is None:
                    match = by_id.get(str(row.get("title") or "")[:120])
                if match is not None:
                    row = dict(row)
                    if not row.get("model_score_human_corrected"):
                        row["model_score"] = match.get("relevance_score")
                        row["model_quote"] = match.get("relevance_quote", "")
                updated.append(row)
            self._populate_human_relevance_table(updated)
            self._persist_human_relevance_model_scores(updated)
        except Exception as exc:
            self._log(f"Could not refresh human queue model scores: {exc}", "ERROR")

    def _persist_human_relevance_model_scores(
        self, rows: list[dict]
    ) -> None:
        """Write refreshed model scores back to the saved review artifact.

        Keeps the on-disk queue in step with the latest scoring run so a
        reopened project shows the same Model Score column as the table.
        """
        if not self._human_review_artifact:
            return
        out_path = self._human_relevance_artifact_path()
        if not out_path.exists():
            return
        self._human_review_artifact["records"] = rows
        try:
            atomic_write_json(out_path, self._human_review_artifact)
        except OSError as exc:
            self._log(f"Could not save refreshed model scores: {exc}", "ERROR")

    def _compare_human_relevance_review(self) -> None:
        from core.io import atomic_write_json
        from process.human_relevance_review import compare_human_to_model

        if not self._human_review_artifact:
            saved = self._human_relevance_artifact_path()
            if saved.exists():
                import json as _json
                with open(saved, encoding="utf-8") as f:
                    self._human_review_artifact = _json.load(f)
                try:
                    self._loaded_human_review_path = saved.resolve()
                except OSError:
                    self._loaded_human_review_path = saved
                self._populate_human_relevance_table(
                    (self._human_review_artifact or {}).get("records", [])
                )
            else:
                self._log("No human relevance queue to compare. Build or save one first.", "ERROR")
                return

        scored_path = self._scored_relevance_path()
        if scored_path is None:
            self._log("No model scored output found yet. Run Score & Filter first.", "ERROR")
            return
        try:
            rows = self._collect_human_relevance_rows()
            scored = self._load_json_list(scored_path)
            summary = compare_human_to_model(
                rows,
                scored,
                threshold=self.score_threshold.value(),
            )
            out_path = self._human_relevance_summary_path()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(out_path, summary)
            self._refresh_human_relevance_model_scores()
            self.human_review_status.setText(
                "Human vs model: "
                f"{summary.get('human_verified', 0)} verified, "
                f"{summary.get('matched', 0)} matched, "
                f"threshold agreement {summary.get('threshold_agreement_pct', 0)}%, "
                f"wκ={summary.get('score_weighted_kappa', 0)}"
            )
            self._log(
                "Human relevance agreement: "
                f"verified={summary.get('human_verified', 0)}, "
                f"{summary.get('matched', 0)} matched, "
                f"exact={summary.get('exact_agreement_pct', 0)}%, "
                f"threshold={summary.get('threshold_agreement_pct', 0)}%, "
                f"wκ={summary.get('score_weighted_kappa', 0)}"
            )
            self._log(f"  Summary: {out_path}")
        except Exception as exc:
            self._log(f"Could not compare human relevance review: {exc}", "ERROR")
    def _on_stat_extract_full_llm_toggled(self, checked: bool):
        if checked:
            self.stat_extract_llm_context.setChecked(True)
        self.stat_extract_max_contexts.setEnabled(not checked)

    def _on_stat_extract_llm_context_toggled(self, checked: bool):
        if not checked and self.stat_extract_full_llm.isChecked():
            self.stat_extract_full_llm.setChecked(False)

    def _start_statistical_extract(self):
        if not self._pipeline_stage_guard("Stage 2c (Statistical Extract)"):
            return
        self._log("═" * 60)
        self._log("STAGE 2c: STATISTICAL EVIDENCE EXTRACTION")
        self._log("═" * 60)

        cfg = self._apply_config_overrides_for_stage(
            self.stat_extract_provider, self.stat_extract_model
        )

        custom_input = self.stat_extract_input.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            input_path = cfg.claims_dir / "relevant.json"

        if not input_path.exists():
            self._log(
                f"No relevant.json found at {input_path}. "
                "Run Stage 2 (Score) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)

        use_llm = self.stat_extract_llm_context.isChecked()
        full_llm = self.stat_extract_full_llm.isChecked()
        max_contexts = 0 if full_llm else self.stat_extract_max_contexts.value()
        self._log(f"Input: {len(papers)} relevant papers from {input_path}")
        self._log(f"Output: {self.stat_extract_output.text().strip() or 'default'}")
        self._log(f"LLM context check: {use_llm}")
        if use_llm:
            limit_label = "all deterministic candidates" if full_llm else str(max_contexts)
            self._log(f"LLM provider: {cfg.llm_provider} / {cfg.llm_model}")
            self._log(f"LLM statistical extraction contexts: {limit_label}")

        custom_output = self.stat_extract_output.text().strip()
        extra_kw = {
            "use_llm_context_check": use_llm,
            "max_llm_contexts": max_contexts,
            "llm_full_statistical_extraction": full_llm,
        }
        if custom_input:
            extra_kw["input_path"] = custom_input
        if custom_output:
            extra_kw["output_path"] = custom_output

        self.stat_extract_btn.setEnabled(False)
        self.stat_extract_cancel_btn.setEnabled(True)
        self.stat_extract_progress.setVisible(True)
        self.stat_extract_progress.setRange(0, 0)
        self.stat_extract_status.setText("Running statistical extraction...")
        self._active_stage_run_btn = self.stat_extract_btn
        self._active_stage_cancel_btn = self.stat_extract_cancel_btn
        self._active_stage_progress = self.stat_extract_progress
        self._active_stage_status = self.stat_extract_status

        self._pipeline_worker = PipelineStageWorker(
            stage_name="statistical_extract",
            config=cfg,
            **extra_kw,
        )
        self._pipeline_worker.progress.connect(self._on_statistical_extract_progress)
        self._pipeline_worker.finished.connect(
            lambda name, result: self._on_statistical_extract_done(result)
        )
        self._pipeline_worker.error.connect(
            lambda name, err: self._on_statistical_extract_error(err)
        )
        self._pipeline_worker.cancelled.connect(
            lambda: self._on_statistical_extract_error("Statistical extraction cancelled")
        )
        self._pipeline_worker.start()

    def _on_statistical_extract_progress(self, msg: str):
        self.stat_extract_status.setText(msg)
        self._log(msg)

    def _on_statistical_extract_done(self, result):
        self.stat_extract_btn.setEnabled(True)
        self.stat_extract_cancel_btn.setEnabled(False)
        self.stat_extract_progress.setVisible(False)
        self.stat_extract_progress.setRange(0, 100)
        if isinstance(result, dict):
            summary = result.get("summary") or {}
            self._log(
                "Statistical extraction complete: "
                f"{summary.get('results', 0)} results, "
                f"{summary.get('llm_reviewed', 0)} LLM-reviewed, "
                f"{summary.get('human_review_items', 0)} review items"
            )
            if result.get("output_path"):
                self._log(f"  Extraction artifact: {result['output_path']}")
            meta_input_path = result.get("meta_analysis_input_path")
            if meta_input_path:
                self._log(f"  Meta-analysis input: {meta_input_path}")
            metrics_path = result.get("metrics_path")
            if metrics_path and metrics_path != meta_input_path:
                self._log(f"  Legacy metrics mirror: {metrics_path}")
            self.stat_extract_status.setText("Statistical extraction complete")
        else:
            self._log(f"Statistical extraction result: {result}")
            self.stat_extract_status.setText("Statistical extraction complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)

    def _on_statistical_extract_error(self, err):
        self._log(f"STATISTICAL EXTRACTION ERROR: {err}", "ERROR")
        self.stat_extract_btn.setEnabled(True)
        self.stat_extract_cancel_btn.setEnabled(False)
        self.stat_extract_progress.setVisible(False)
        self.stat_extract_progress.setRange(0, 100)
        self.stat_extract_status.setText(str(err))

    def _connect_stage_worker(self, on_done, on_error, on_cancelled=None):
        """Wire the standard worker signals for a stage launcher.

        Connects progress (to the log), finished, error, and optionally
        cancelled. Every stage launcher used to repeat this block verbatim.
        """
        self._pipeline_worker.progress.connect(lambda msg: self._log(msg))
        self._pipeline_worker.finished.connect(
            lambda name, result: on_done(result))
        self._pipeline_worker.error.connect(
            lambda name, err: on_error(err))
        if on_cancelled is not None:
            self._pipeline_worker.cancelled.connect(on_cancelled)


    def _start_sysys_score(self):
        if not self._pipeline_stage_guard("Stage 2 (Score)"):
            return
        self._log("═" * 60)
        self._log("STAGE 2: RELEVANCE SCORING")
        self._log("═" * 60)

        cfg = self._apply_config_overrides_for_stage(self.score_provider, self.score_model)

        # Check corpus exists
        corpus_path = cfg.corpus_dir / "corpus.json"
        custom_path = self.score_corpus_path.text().strip()
        if custom_path:
            from pathlib import Path as _Path
            corpus_path = _Path(custom_path)
        if not corpus_path.exists():
            self._log("No corpus found. Run Stage 1 (Search) first.", "ERROR")
            return

        import json as _json
        with open(corpus_path, encoding="utf-8") as f:
            corpus = _json.load(f)

        cfg.relevance_threshold = self.score_threshold.value()
        cfg.score_batch_size = self.score_batch.value()

        # If the user hasn't opted into advanced LLM options, lock Stage 2 to
        # the bespoke BRT-BERT model and ignore the (hidden) provider combo.
        simple_mode = not (
            hasattr(self, "score_advanced_box") and not self.score_advanced_box.isHidden()
        )
        if simple_mode:
            selected_provider = "brt-bert"
            cfg.relevance_model = "brt-bert"
        else:
            selected_provider = self.score_provider.currentText().strip()
            if selected_provider in ("brt-bert", "housecatbert-scibert-releevance"):
                score_model = self.score_model.text().strip()
                cfg.relevance_model = (
                    score_model
                    if score_model.lower().startswith(("brt", "housecatbert"))
                    else selected_provider
                )
            else:
                cfg.relevance_model = ""

        self._log(f"Corpus: {len(corpus)} papers from {corpus_path}")
        self._log(f"Provider: {selected_provider}")
        self._log(f"Model: {cfg.relevance_model if cfg.relevance_model else cfg.llm_model}")
        self._log(f"Threshold: {self.score_threshold.value()}")
        self._log(f"Batch size: {self.score_batch.value()}")
        self._log(f"Workers: {cfg.concurrent_workers}")


        self.score_run_btn.setEnabled(False)
        self.score_cancel_btn.setEnabled(True)
        self._active_stage_run_btn = self.score_run_btn
        self._active_stage_cancel_btn = self.score_cancel_btn

        self._pipeline_worker = PipelineStageWorker(
            stage_name="score",
            config=cfg,
            corpus_path=corpus_path,
            threshold=cfg.relevance_threshold,
            workload=self.score_workload.currentText().lower(),
        )
        self._connect_stage_worker(self._on_score_done, self._on_score_error)
        self._set_pipeline_stage(2, "running")
        self._pipeline_worker.start()

    def _on_score_done(self, result):
        self.score_run_btn.setEnabled(True)
        self.score_cancel_btn.setEnabled(False)
        if isinstance(result, list):
            self._log(f"Scoring complete: {len(result)} relevant papers")
            if hasattr(self, "_refresh_human_relevance_model_scores"):
                self._refresh_human_relevance_model_scores()
            if hasattr(self, "_set_corpus_stage_relevance"):
                self._set_corpus_stage_relevance("stage2", result)
            # Feature D + F
            self._update_glance_relevant(len(result))
            self._update_glance_stage("Score done")
            self._set_tab_badge(1, "Score && Filter", len(result))
            n_rel = len(result)
            rel_label = "paper" if n_rel == 1 else "papers"
            self._set_pipeline_stage(2, "done", summary=f"{n_rel} relevant {rel_label}")
            if self._run_all_active:
                self._run_all_stages_advance()
        else:
            self._log(f"Scoring result: {result}")
            self._set_pipeline_stage(2, "failed", summary="Unexpected score result")
            if self._run_all_active:
                self._run_all_active = False
                self._run_all_queue.clear()
                self._log("RUN ALL aborted at stage 2 (unexpected score result).", "ERROR")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 2 score complete")

    def _on_score_error(self, err):
        self._log(f"SCORING ERROR: {err}", "ERROR")
        self._set_pipeline_stage(2, "failed", summary=self._pipeline_error_summary(err))
        self.score_run_btn.setEnabled(True)
        self.score_cancel_btn.setEnabled(False)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 2 (score error).", "ERROR")

    def _start_sysys_corpus_fulltext(self, auto_started: bool = False,
                                       triggered_by_fetch: bool = False):
        """Enrich the Stage 1 corpus with full text (download step).

        Used both by the "Download full text" auto-path after a search and by
        the "Fetch Articles" button (``triggered_by_fetch=True``), which is a
        download-only action over the existing corpus.
        """
        if not self._pipeline_stage_guard("Corpus full-text enrichment"):
            return
        self._sync_fulltext_controls_to_config()
        self._log("═" * 60)
        self._log("CORPUS FULL-TEXT ENRICHMENT")
        self._log("═" * 60)
        if not self._pipeline_corpus:
            self._log("No corpus loaded. Run Stage 1 (Search) first.", "ERROR")
            self._update_search_workflow_buttons()
            return
        self._log(f"Corpus: {len(self._pipeline_corpus)} papers")
        if auto_started:
            self._log("Download full text is enabled; enriching Stage 1 corpus now.")

        cfg = self.pipeline_config
        corpus_path = cfg.corpus_dir / "corpus.json"
        output_path = cfg.corpus_dir / "corpus_fulltext.json"
        if not corpus_path.exists():
            self._log(f"No Stage 1 corpus found at {corpus_path}", "ERROR")
            self._update_search_workflow_buttons()
            return

        self._log(f"Input: {corpus_path}")
        self._log(f"Output: {output_path}")
        self._log(f"Archive: {cfg.data_dir / 'fulltext_corpus' / 'source_archive'}")
        if cfg.paperqa_index_enabled:
            self._log(
                f"PaperQA store: {cfg.data_dir / 'fulltext_corpus' / 'paperqa'} "
                f"(checkpoint batch {cfg.paperqa_index_batch_size})"
            )
        else:
            self._log("PaperQA indexing: disabled")

        from gui.workers import PipelineStageWorker

        # Disable both run buttons while downloading and enable the cancel
        # button matching whichever button started the download.
        self.sysys_search_btn.setEnabled(False)
        if hasattr(self, "fetch_btn"):
            self.fetch_btn.setEnabled(False)
        if triggered_by_fetch and hasattr(self, "cancel_fetch_btn"):
            self.cancel_fetch_btn.setEnabled(True)
            active_cancel_btn = self.cancel_fetch_btn
        else:
            self.sysys_cancel_btn.setEnabled(True)
            active_cancel_btn = self.sysys_cancel_btn
        self.search_progress.setVisible(True)
        self.search_progress.setRange(0, 0)
        self.search_status.setText("Enriching corpus with full text...")
        self.pipeline_state = PipelineState.SEARCHING
        self._active_stage_run_btn = self.sysys_search_btn
        self._active_stage_cancel_btn = active_cancel_btn
        self._active_stage_progress = self.search_progress

        self._pipeline_worker = PipelineStageWorker(
            stage_name="corpus_fulltext",
            config=cfg,
            corpus_path=corpus_path,
            output_path=output_path,
        )
        self._pipeline_worker.progress.connect(self._on_sysys_search_progress)
        self._pipeline_worker.progress_counts.connect(self._on_sysys_search_counts)
        self._pipeline_worker.finished.connect(
            lambda name, result: self._on_sysys_corpus_fulltext_done(result)
        )
        self._pipeline_worker.error.connect(
            lambda name, err: self._on_sysys_corpus_fulltext_error(err)
        )
        self._pipeline_worker.cancelled.connect(
            lambda: self._on_sysys_corpus_fulltext_error("Corpus full-text enrichment cancelled")
        )
        self._pipeline_worker.start()

    def _on_sysys_corpus_fulltext_done(self, result):
        self.sysys_search_btn.setEnabled(True)
        self.sysys_cancel_btn.setEnabled(False)
        if hasattr(self, "cancel_fetch_btn"):
            self.cancel_fetch_btn.setEnabled(False)
        self.search_progress.setVisible(False)
        self.pipeline_state = PipelineState.IDLE

        if not isinstance(result, dict):
            self._log(f"Corpus full-text result: {result}")
            self._update_search_workflow_buttons()
            return

        corpus_path = result.get("corpus_path", "")
        fulltext_count = int(result.get("fulltext_count", 0) or 0)
        paper_count = int(result.get("paper_count", 0) or 0)
        self._log(
            f"Corpus full-text enrichment complete: {fulltext_count}/{paper_count} papers"
        )
        if corpus_path:
            self._log(f"  Enriched corpus: {corpus_path}")
            try:
                from pathlib import Path as _Path
                import json as _json

                enriched_path = _Path(corpus_path)
                with enriched_path.open(encoding="utf-8") as f:
                    self._pipeline_corpus = _json.load(f)
                self._populate_corpus_browser(self._pipeline_corpus)
                self.score_corpus_path.setText(str(enriched_path))
            except Exception as exc:
                self._log(f"Could not load enriched corpus preview: {exc}", "WARNING")
        archive_dir = result.get("archive_dir", "")
        if archive_dir:
            self._log(f"  Compressed source archive: {archive_dir}")
        paperqa = result.get("paperqa") or {}
        if paperqa.get("available"):
            self._log(
                "  PaperQA indexed "
                f"{paperqa.get('added', 0)} new texts "
                f"({paperqa.get('indexed_total', 0)} total): "
                f"{paperqa.get('docs_path', '')}"
            )
        else:
            reason = (
                paperqa.get("reason")
                or paperqa.get("error")
                or "paper-qa not installed"
            )
            self._log(
                f"  PaperQA indexing skipped: {reason}",
                "INFO" if reason == "disabled" else "WARNING",
            )
        self._keyword_analysis_ready = bool(self._pipeline_corpus or self.articles)
        self._update_search_workflow_buttons()
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("corpus fulltext complete")

    def _on_sysys_corpus_fulltext_error(self, err: str):
        self._log(f"CORPUS FULLTEXT ERROR: {err}", "ERROR")
        self.sysys_search_btn.setEnabled(True)
        self.sysys_cancel_btn.setEnabled(False)
        if hasattr(self, "cancel_fetch_btn"):
            self.cancel_fetch_btn.setEnabled(False)
        self.search_progress.setVisible(False)
        self.pipeline_state = PipelineState.IDLE
        self._update_search_workflow_buttons()

    def _start_extract(self):
        if not self._pipeline_stage_guard("Stage 3 (Extract)"):
            return
        self._log("═" * 60)
        self._log("STAGE 3: CLAIM EXTRACTION")
        self._log("═" * 60)

        cfg = self._apply_config_overrides_for_stage(
            self.extract_provider, self.extract_model)

        # Resolve input path
        custom_input = self.extract_corpus.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            input_path = cfg.claims_dir / "relevant.json"

        if not input_path.exists():
            self._log(
                f"No relevant.json found at {input_path}. "
                "Run Stage 2 (Score) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)

        self._log(f"Input: {len(papers)} relevant papers from {input_path}")
        self._log(f"Provider: {cfg.llm_provider}")
        self._log(f"Model: {cfg.llm_model}")
        self._log(f"Max tokens: {self.extract_max_tokens.value()}")
        self._log(f"Thinking: {self.extract_thinking.isChecked()}")

        # Resolve output path
        custom_output = self.extract_output.text().strip()
        extra_kw = {
            "max_tokens": self.extract_max_tokens.value(),
            "use_thinking": self.extract_thinking.isChecked(),
        }
        if custom_input:
            extra_kw["input_path"] = custom_input
        if custom_output:
            extra_kw["output_path"] = custom_output

        self.extract_btn.setEnabled(False)
        self.extract_cancel_btn.setEnabled(True)
        self.extract_validate_progress.setVisible(True)
        self._active_stage_run_btn = self.extract_btn
        self._active_stage_cancel_btn = self.extract_cancel_btn
        self._active_stage_progress = self.extract_validate_progress

        self._pipeline_worker = PipelineStageWorker(
            stage_name="extract",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(self._on_extract_done, self._on_extract_error)
        self._set_pipeline_stage(3, "running")
        self._pipeline_worker.start()

    def _on_extract_done(self, result):
        self.extract_btn.setEnabled(True)
        self.extract_cancel_btn.setEnabled(False)
        self.extract_validate_progress.setVisible(False)
        if isinstance(result, dict):
            n = result.get("extracted", 0)
            n_f = result.get("with_findings", 0)
            n_m = result.get("with_mechanisms", 0)
            self._log(
                f"Extraction complete: {n} papers — "
                f"{n_f} with findings, {n_m} with mechanisms")
            self._set_pipeline_stage(
                3, "done",
                summary=f"{n} papers extracted ({n_f} with findings)",
            )
        else:
            self._log(f"Extraction result: {result}")
            self._set_pipeline_stage(3, "done", summary="Extraction complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 3 extract complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_extract_error(self, err):
        self._log(f"EXTRACTION ERROR: {err}", "ERROR")
        self._set_pipeline_stage(3, "failed", summary=self._pipeline_error_summary(err))
        self.extract_btn.setEnabled(True)
        self.extract_cancel_btn.setEnabled(False)
        self.extract_validate_progress.setVisible(False)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 3 (extract error).", "ERROR")

    def _start_validate(self):
        if not self._pipeline_stage_guard("Stage 4 (Validate)"):
            return
        self._log("═" * 60)
        self._log("STAGE 4: METHOD REVIEW / VALIDATION")
        self._log("═" * 60)

        cfg = self._apply_config_overrides_for_stage(
            self.validate_provider, self.validate_model)

        # Resolve input path
        custom_input = self.validate_claims.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            input_path = cfg.claims_dir / "relevant.json"

        if not input_path.exists():
            self._log(
                f"No input found at {input_path}. "
                "Run Stage 2 (Score) or Stage 3 (Extract) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)

        self._log(f"Input: {len(papers)} papers from {input_path}")
        self._log(f"Provider: {cfg.llm_provider}")
        self._log(f"Model: {cfg.llm_model}")

        # Build kwargs
        custom_output = self.validate_output.text().strip()
        extra_kw = {}
        if custom_input:
            extra_kw["input_path"] = custom_input
        if custom_output:
            extra_kw["output_path"] = custom_output

        self.validate_btn.setEnabled(False)
        self.validate_cancel_btn.setEnabled(True)
        self.extract_validate_progress.setVisible(True)
        self._active_stage_run_btn = self.validate_btn
        self._active_stage_cancel_btn = self.validate_cancel_btn
        self._active_stage_progress = self.extract_validate_progress

        self._pipeline_worker = PipelineStageWorker(
            stage_name="validate",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(self._on_validate_done, self._on_validate_error)
        self._set_pipeline_stage(4, "running")
        self._pipeline_worker.start()

    def _on_validate_done(self, result):
        self.validate_btn.setEnabled(True)
        self.validate_cancel_btn.setEnabled(False)
        self.extract_validate_progress.setVisible(False)
        if isinstance(result, dict):
            n = result.get("validated", 0)
            flagged = result.get("flagged", 0)
            self._log(
                f"Method review complete: {n} papers validated, "
                f"{flagged} flagged for manual review")
            summary = f"{n} profiles"
            if flagged:
                summary = f"{summary}, {flagged} flagged"
            self._set_pipeline_stage(4, "done", summary=summary)
        else:
            self._log(f"Validation result: {result}")
            self._set_pipeline_stage(4, "done", summary="Validation complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 4 validate complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_validate_error(self, err):
        self._log(f"VALIDATION ERROR: {err}", "ERROR")
        self._set_pipeline_stage(4, "failed", summary=self._pipeline_error_summary(err))
        self.validate_btn.setEnabled(True)
        self.validate_cancel_btn.setEnabled(False)
        self.extract_validate_progress.setVisible(False)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 4 (validate error).", "ERROR")

    def _start_audit(self):
        if not self._pipeline_stage_guard("Stage 5 (Audit)"):
            return
        self._log("═" * 60)
        self._log("STAGE 5: SPAN AUDIT & QUALITY FILTERING")
        self._log("═" * 60)

        import copy as _copy
        cfg = _copy.deepcopy(self.pipeline_config)

        # Resolve input — claims.json from Stage 3
        custom_input = self.audit_input.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            input_path = cfg.claims_dir / "claims.json"

        if not input_path.exists():
            self._log(
                f"No claims found at {input_path}. "
                "Run Stage 3 (Extract) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)
        self._log(f"Input: {len(papers)} papers from {input_path}")

        # Read threshold slider
        threshold = self.audit_threshold.value() / 100.0
        self._log(f"Similarity threshold: {threshold:.2f}")

        # Build kwargs
        extra_kw = {}
        if custom_input:
            extra_kw["claims_path"] = custom_input
        custom_output = self.audit_output.text().strip()
        if custom_output:
            extra_kw["output_path"] = custom_output

        self.audit_btn.setEnabled(False)
        self._active_stage_run_btn = self.audit_btn
        self._active_stage_cancel_btn = None

        self._pipeline_worker = PipelineStageWorker(
            stage_name="audit",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(self._on_audit_done, self._on_audit_error)
        self._set_pipeline_stage(5, "running")
        self._pipeline_worker.start()

    def _on_audit_done(self, result):
        self.audit_btn.setEnabled(True)
        if isinstance(result, dict):
            total = result.get("total", 0)
            kept = result.get("filtered", 0)
            removed = result.get("removed", 0)
            rescued = result.get("reclassified", 0)
            self._log(
                f"Span audit complete: {total} papers in, "
                f"{kept} kept, {removed} removed, {rescued} rescued")
            out = result.get("output_path", "")
            if out:
                self._log(f"  Output: {out}")
            self._set_pipeline_stage(5, "done", summary=f"{kept} kept, {removed} removed")
        else:
            self._log(f"Audit result: {result}")
            self._set_pipeline_stage(5, "done", summary="Audit complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 5 audit complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_audit_error(self, err):
        self._log(f"AUDIT ERROR: {err}", "ERROR")
        self._set_pipeline_stage(5, "failed", summary=self._pipeline_error_summary(err))
        self.audit_btn.setEnabled(True)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 5 (audit error).", "ERROR")

    def _start_synthesize(self):
        if not self._pipeline_stage_guard("Stage 6 (Synthesize)"):
            return
        self._log("═" * 60)
        self._log("STAGE 6: SYNTHESIS (CLUSTER · NARRATE · ABSTRACT)")
        self._log("═" * 60)

        cfg = self._apply_config_overrides_for_stage(
            self.synth_provider, self.synth_model)

        # Resolve input — prefer claims_filtered.json, then claims.json
        custom_input = self.synth_input.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            # Prefer filtered output from Stage 5, fall back to Stage 3
            # (resolve_claims_path also falls back when the filtered file
            # exists but is empty)
            from process.paths import resolve_claims_path
            input_path = resolve_claims_path(cfg)

        if not input_path.exists():
            self._log(
                f"No claims found at {input_path}. "
                "Run Stage 3 (Extract) or Stage 5 (Audit) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)
        self._log(f"Input: {len(papers)} papers from {input_path}")

        auto_clusters = self.synth_auto_clusters.isChecked()
        n_clusters = 0 if auto_clusters else self.synth_min_cluster.value()
        max_tokens = self.synth_max_tokens.value()
        if auto_clusters:
            self._log(f"Clusters: auto (silhouette sweep), max tokens: {max_tokens}")
        else:
            self._log(f"Clusters: up to {n_clusters}, max tokens: {max_tokens}")
        self._log(f"Provider: {cfg.llm_provider}, Model: {cfg.llm_model}")

        # Build kwargs — pass the resolved input so the stage uses exactly
        # the file validated above
        extra_kw: dict = {
            "n_clusters": n_clusters,
            "max_tokens": max_tokens,
            "claims_path": str(input_path),
        }
        custom_output = self.synth_output.text().strip()
        if custom_output:
            extra_kw["output_path"] = custom_output

        self.synth_btn.setEnabled(False)
        self._active_stage_run_btn = self.synth_btn
        self._active_stage_cancel_btn = None

        self._pipeline_worker = PipelineStageWorker(
            stage_name="synthesize",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(self._on_synthesize_done, self._on_synthesize_error)
        self._set_pipeline_stage(6, "running")
        self._pipeline_worker.start()

    def _on_synthesize_done(self, result):
        self.synth_btn.setEnabled(True)
        if isinstance(result, dict):
            total = result.get("total_papers", 0)
            n_clust = result.get("n_clusters", 0)
            n_narr = result.get("narratives_count", 0)
            abs_len = result.get("abstract_length", 0)
            self._log(
                f"Synthesis complete: {total} papers, {n_clust} clusters, "
                f"{n_narr} narratives, abstract {abs_len} chars")
            k_sel = result.get("k_selection") or {}
            if k_sel.get("mode") == "auto":
                sweep = k_sel.get("silhouette_scores", [])
                if sweep:
                    self._log("  Silhouette sweep: " + ", ".join(
                        f"k={s['k']}:{s['score']:.3f}" for s in sweep))
                self._log(
                    f"  Auto-selected k={k_sel.get('chosen_k')} by silhouette score")
            abstract_path = result.get("abstract_path", "")
            if abstract_path:
                self._log(f"  Abstract: {abstract_path}")
                try:
                    with open(abstract_path, encoding="utf-8") as f:
                        abstract_text = f.read()
                    self.synth_narrative.setPlainText(abstract_text)
                except Exception as exc:
                    logger.warning("Could not load abstract for display: %s", exc)
                # Auto-open the abstract in a pop-out window
                self._open_abstract_window(Path(abstract_path))
            clusters_path = result.get("clusters_path", "")
            if clusters_path:
                try:
                    import json as _cj
                    with open(clusters_path, encoding="utf-8") as f:
                        clusters_data = _cj.load(f)
                    clusters_list = clusters_data if isinstance(clusters_data, list) else \
                        clusters_data.get("clusters", [])
                    self.cluster_table.setRowCount(len(clusters_list))
                    for row, cl in enumerate(clusters_list):
                        label = cl.get("label") or cl.get("name") or f"Cluster {row + 1}"
                        n_papers = cl.get("paper_count", 0) or len(cl.get("papers", []))
                        terms = ", ".join(
                            (cl.get("mechanisms", []) + cl.get("conditions", []))[:5]
                        )
                        self.cluster_table.setItem(row, 0, QTableWidgetItem(str(label)))
                        self.cluster_table.setItem(row, 1, QTableWidgetItem(str(n_papers)))
                        self.cluster_table.setItem(row, 2, QTableWidgetItem(terms))
                    self.cluster_table.resizeColumnsToContents()
                except Exception as exc:
                    logger.warning("Could not populate cluster table: %s", exc)
            if clusters_path:
                self._log(f"  Clusters: {clusters_path}")
            narratives_path = result.get("narratives_path", "")
            if narratives_path:
                self._log(f"  Narratives: {narratives_path}")
            explanation_md = result.get("hypothesis_explanation_md_path", "")
            if explanation_md:
                self._log(f"  Why This Hypothesis: {explanation_md}")
                self._display_hypothesis_explanation(Path(explanation_md))
            explanation_json = result.get("hypothesis_explanation_path", "")
            if explanation_json:
                self._log(f"  Explanation JSON: {explanation_json}")
            evidence_network = result.get("evidence_network_path", "")
            if evidence_network:
                self._log(f"  Evidence Network: {evidence_network}")
                self._load_evidence_network_path(Path(evidence_network))
            sources_md = result.get("synthesis_sources_md_path", "")
            if sources_md:
                self._log(f"  Source Citations: {sources_md}")
                self._display_synthesis_sources(Path(sources_md))
            sources_bib = result.get("synthesis_sources_bib_path", "")
            if sources_bib:
                self._log(f"  Source BibTeX: {sources_bib}")
            sources_json = result.get("synthesis_sources_path", "")
            if sources_json:
                self._log(f"  Source Map JSON: {sources_json}")
            n_clust = result.get("n_clusters", 0)
            synth_parts = []
            if n_clust:
                synth_parts.append(f"{n_clust} clusters")
            if result.get("abstract_path") or result.get("abstract_length"):
                synth_parts.append("abstract saved")
            self._set_pipeline_stage(
                6, "done",
                summary="; ".join(synth_parts) if synth_parts else "Synthesis complete",
            )
            # Land on the Abstract sub-tab so the result is front and center
            self._show_synth_output_subtab(0)
        else:
            self._log(f"Synthesis result: {result}")
            self._set_pipeline_stage(6, "done", summary="Synthesis complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 6 synthesize complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_synthesize_error(self, err):
        self._log(f"SYNTHESIS ERROR: {err}", "ERROR")
        self._set_pipeline_stage(6, "failed", summary=self._pipeline_error_summary(err))
        self.synth_btn.setEnabled(True)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 6 (synthesize error).", "ERROR")

    def _display_hypothesis_explanation(self, path: Path) -> bool:
        """Load hypothesis_explanation.md into the synthesis detail panel."""
        try:
            text = Path(path).read_text(encoding="utf-8")
            if hasattr(self.hypothesis_explanation, "setMarkdown"):
                self.hypothesis_explanation.setMarkdown(text)
            else:
                self.hypothesis_explanation.setPlainText(text)
            self._log(f"Hypothesis explanation loaded: {path}")
            return True
        except Exception as exc:
            self._log(f"Could not load hypothesis explanation: {exc}", "ERROR")
            return False

    def _show_synth_output_subtab(self, index: int) -> None:
        """Bring the requested Synthesis output sub-tab to the front."""
        tabs = getattr(self, "synth_output_tabs", None)
        if tabs is not None and 0 <= index < tabs.count():
            tabs.setCurrentIndex(index)

    def _load_hypothesis_explanation(self):
        """Load the latest deterministic hypothesis explanation Markdown."""
        default_path = self.pipeline_config.clusters_dir / "hypothesis_explanation.md"
        path = default_path
        if not path.exists():
            chosen, _ = QFileDialog.getOpenFileName(
                self,
                "Load Hypothesis Explanation",
                str(self.pipeline_config.clusters_dir),
                "Markdown files (*.md);;All files (*)",
            )
            if not chosen:
                return
            path = Path(chosen)
        self._display_hypothesis_explanation(path)
        self._show_synth_output_subtab(1)
        self._open_text_result_window("Why This Hypothesis", path)

    def _display_synthesis_sources(self, path: Path) -> bool:
        """Load synthesis_sources.md into the dedicated Source Citations view.

        Uses ``synth_sources_view`` so the abstract in ``synth_narrative``
        is never overwritten.
        """
        view = getattr(self, "synth_sources_view", None) or self.synth_narrative
        try:
            text = Path(path).read_text(encoding="utf-8")
            if hasattr(view, "setMarkdown"):
                view.setMarkdown(text)
            else:
                view.setPlainText(text)
            self._log(f"Synthesis source citations loaded: {path}")
            return True
        except Exception as exc:
            self._log(f"Could not load synthesis source citations: {exc}", "ERROR")
            return False

    def _load_synthesis_sources(self):
        """Load the latest synthesis source citation Markdown."""
        default_path = self.pipeline_config.clusters_dir / "synthesis_sources.md"
        path = default_path
        if not path.exists():
            chosen, _ = QFileDialog.getOpenFileName(
                self,
                "Load Synthesis Source Citations",
                str(self.pipeline_config.clusters_dir),
                "Markdown files (*.md);;All files (*)",
            )
            if not chosen:
                return
            path = Path(chosen)
        self._display_synthesis_sources(path)
        self._show_synth_output_subtab(2)
        self._open_text_result_window("Source Citations", path)

    def _load_evidence_network_path(self, path: Path) -> bool:
        """Load generalized evidence_network.json into the network widgets."""
        loaded = False
        try:
            import json as _json
            data = _json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"Could not load evidence network: {exc}", "ERROR")
            return False
        if hasattr(self, "neuro_network_widget"):
            self.neuro_network_widget.set_network_data(data)
            loaded = True
        if hasattr(self, "synth_network_widget"):
            self.synth_network_widget.set_network_data(data)
            loaded = True
        if loaded:
            if hasattr(self, "neuro_export_graph_btn"):
                self.neuro_export_graph_btn.setEnabled(True)
            self._log(f"Evidence network loaded: {path}")
        return loaded

    def _load_evidence_network(self):
        """Load the latest generalized evidence network JSON."""
        default_path = self.pipeline_config.clusters_dir / "evidence_network.json"
        path = default_path
        if not path.exists():
            chosen, _ = QFileDialog.getOpenFileName(
                self,
                "Load Evidence Network",
                str(self.pipeline_config.clusters_dir),
                "JSON files (*.json);;All files (*)",
            )
            if not chosen:
                return
            path = Path(chosen)
        self._load_evidence_network_path(path)
        self._show_synth_output_subtab(3)
        self._open_evidence_network_window(path)

    # ── Pop-out result viewer windows ──────────────────────────────────

    def _open_text_result_window(
        self,
        title: str,
        path: Path,
        render_markdown: bool = True,
    ) -> Optional["object"]:
        """Open a TextResultWindow showing the contents of *path*."""
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            self._log(f"Could not open result window for {path}: {exc}", "ERROR")
            return None
        from gui.result_viewer import TextResultWindow

        win = TextResultWindow(title, source_path=path, parent=self)
        win.set_text(text, render_markdown=render_markdown)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.show()
        win.raise_()
        win.activateWindow()
        # Keep a reference so the window isn't garbage-collected
        if not hasattr(self, "_result_windows"):
            self._result_windows = []
        self._result_windows.append(win)
        win.destroyed.connect(lambda _w=None, win=win: self._result_windows.remove(win)
                              if win in self._result_windows else None)
        return win

    def _open_abstract_window(self, abstract_path: Path) -> None:
        """Open a TextResultWindow showing the synthesis abstract."""
        self._open_text_result_window(
            "Synthesis Abstract", abstract_path, render_markdown=True
        )

    def _open_evidence_network_window(self, path: Path) -> None:
        """Open a NetworkResultWindow showing the evidence network graph + JSON."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"Could not open evidence network window: {exc}", "ERROR")
            return
        from gui.result_viewer import NetworkResultWindow

        win = NetworkResultWindow(
            "Evidence Network", source_path=path, parent=self
        )
        win.set_network_data(data)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.show()
        win.raise_()
        win.activateWindow()
        if not hasattr(self, "_result_windows"):
            self._result_windows = []
        self._result_windows.append(win)
        win.destroyed.connect(lambda _w=None, win=win: self._result_windows.remove(win)
                              if win in self._result_windows else None)

    def _start_sysys_fulltext(self):
        if not self._pipeline_stage_guard("Stage 7 (Fulltext)"):
            return
        self._sync_fulltext_controls_to_config()
        self._log("═" * 60)
        self._log("STAGE 7: FULL-TEXT RETRIEVAL & METRIC EXTRACTION")
        self._log("═" * 60)

        cfg = self.pipeline_config

        # Resolve input corpus path
        custom_input = self.sysys_ft_corpus.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            # Default: claims from Stage 3/5 (resolve_claims_path falls
            # back to claims.json when the filtered file is empty)
            from process.paths import resolve_claims_path
            input_path = resolve_claims_path(cfg)

        if not input_path.exists():
            self._log(
                f"No claims found at {input_path}. "
                "Run Stage 3 (Extract) or Stage 5 (Audit) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)
        self._log(f"Input: {len(papers)} papers from {input_path}")

        workers = self.sysys_ft_workers.value()
        email = getattr(cfg, "pubmed_email", "") or "research@pipeline.dev"
        self._log(f"Workers: {workers}, Email: {email}")
        if cfg.fulltext_streaming:
            self._log("Streaming: enabled (resumable JSONL checkpoints)")
        else:
            self._log(
                f"Streaming: disabled (buffered batch {cfg.fulltext_download_batch_size})"
            )

        extra_kw: dict = {
            "workers": workers,
            "email": email,
            "resume": self.sysys_ft_resume.isChecked(),
            # Pass the validated input so the stage uses the same file
            "claims_path": str(input_path),
        }

        self.sysys_ft_btn.setEnabled(False)
        self.sysys_ft_cancel_btn.setEnabled(True)
        self._active_stage_run_btn = self.sysys_ft_btn
        self._active_stage_cancel_btn = self.sysys_ft_cancel_btn
        if self._project_path is not None:
            self._save_project()

        from gui.workers import PipelineStageWorker
        self._pipeline_worker = PipelineStageWorker(
            stage_name="fulltext",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(
            self._on_fulltext_done, self._on_fulltext_error,
            on_cancelled=self._on_fulltext_cancelled)
        self._set_pipeline_stage(7, "running")
        self._pipeline_worker.start()

    def _on_fulltext_cancelled(self):
        self.sysys_ft_btn.setEnabled(True)
        self.sysys_ft_cancel_btn.setEnabled(False)
        self._run_all_active = False
        self._run_all_queue.clear()
        self._set_pipeline_stage(7, "pending", summary="Cancelled; checkpoints preserved")
        self._log("Fulltext cancelled; completed paper checkpoints preserved.")

    def _on_fulltext_done(self, result):
        self.sysys_ft_btn.setEnabled(True)
        self.sysys_ft_cancel_btn.setEnabled(False)
        if isinstance(result, dict):
            targets = result.get("targets", 0)
            oa = result.get("oa_papers", 0)
            downloaded = result.get("downloaded", 0)
            metrics = result.get("metrics_extracted", 0)
            self._log(
                f"Fulltext complete: {targets} targets identified, "
                f"{oa} open access, {downloaded} downloaded, "
                f"{metrics} with metrics extracted")
            resumed = int(result.get("resumed", 0) or 0)
            remaining = int(result.get("remaining", 0) or 0)
            if resumed:
                self._log(f"  Resumed {resumed} completed papers from JSONL")
            if remaining:
                self._log(
                    f"  {remaining} papers remain eligible for retry on the next run",
                    "WARNING",
                )
            out = result.get("output_path", "")
            if out:
                self._log(f"  Output: {out}")
            self._set_pipeline_stage(
                7, "done",
                summary=f"{downloaded} downloaded, {metrics} with metrics",
            )
        else:
            self._log(f"Fulltext result: {result}")
            self._set_pipeline_stage(7, "done", summary="Fulltext complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 7 fulltext complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_fulltext_error(self, err):
        self._log(f"FULLTEXT ERROR: {err}", "ERROR")
        self._set_pipeline_stage(7, "failed", summary=self._pipeline_error_summary(err))
        self.sysys_ft_btn.setEnabled(True)
        self.sysys_ft_cancel_btn.setEnabled(False)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 7 (fulltext error).", "ERROR")

    def _start_interrater(self):
        if not self._pipeline_stage_guard("Stage 8 (Interrater)"):
            return
        self._log("═" * 60)
        self._log("STAGE 8: INTER-RATER RELIABILITY")
        self._log("═" * 60)

        # The interrater_provider/model widgets specify the SECOND rater; the
        # primary rater is whatever produced the scores already in the corpus,
        # i.e. the Stage 2 (Score) settings.  Build the config from Stage 2 so
        # run_interrater can tell the two raters apart — overriding it with the
        # second rater here would make every run look like a matched pair.
        cfg = self._apply_config_overrides_for_stage(
            self.score_provider, self.score_model)

        primary_provider = self.score_provider.currentText().strip()
        primary_model = self.score_model.text().strip()
        second_provider = self.interrater_provider.currentText()
        second_model = self.interrater_model.text().strip()

        # Resolve corpus path — prefer custom path, then standard locations
        custom_input = self.interrater_corpus.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            corpus_path = _Path(custom_input)
        else:
            # resolve_claims_path falls back to claims.json when the
            # filtered file exists but is empty
            from process.paths import resolve_claims_path
            resolved = resolve_claims_path(cfg)
            relevant = cfg.corpus_dir / "relevant.json"
            if resolved.exists():
                corpus_path = resolved
            elif relevant.exists():
                corpus_path = relevant
            else:
                corpus_path = None

        if corpus_path is None or not corpus_path.exists():
            self._log(
                "No scored corpus found. Run Stage 2 (Score) or "
                "Stage 3 (Extract) first.", "ERROR")
            return

        import json as _json
        with open(corpus_path, encoding="utf-8") as f:
            papers = _json.load(f)
        if not papers:
            self._log(
                f"Corpus at {corpus_path} is empty — nothing to score.", "ERROR")
            return

        sample_size = self.interrater_sample.value()
        self._log(f"Input: {len(papers)} papers from {corpus_path}")
        self._log(f"Sample size: {sample_size}")
        self._log(f"Primary: {primary_provider} / {primary_model or 'default'}")
        self._log(f"Second rater: {second_provider} / {second_model or 'default'}")

        extra_kw: dict = {
            "sample_size": sample_size,
            "second_provider": second_provider,
            "second_model": second_model or None,
            "primary_provider": primary_provider or None,
            "primary_model": primary_model or None,
            # Pass the validated corpus so the stage uses the same file
            "claims_path": str(corpus_path),
        }

        self.interrater_btn.setEnabled(False)
        self.interrater_cancel_btn.setEnabled(True)
        self._active_stage_run_btn = self.interrater_btn
        self._active_stage_cancel_btn = self.interrater_cancel_btn

        self._pipeline_worker = PipelineStageWorker(
            stage_name="interrater",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(
            self._on_interrater_done, self._on_interrater_error,
            on_cancelled=self._on_interrater_cancelled)
        self._set_pipeline_stage(8, "running")
        self._pipeline_worker.start()

    def _on_interrater_cancelled(self):
        self.interrater_btn.setEnabled(True)
        self.interrater_cancel_btn.setEnabled(False)
        self._run_all_active = False
        self._run_all_queue.clear()
        self._set_pipeline_stage(8, "pending", summary="Cancelled; checkpoint saved")
        self._log("Interrater cancelled; checkpoint saved — rerun to resume.")

    def _on_interrater_done(self, result):
        self.interrater_btn.setEnabled(True)
        self.interrater_cancel_btn.setEnabled(False)
        if isinstance(result, dict):
            n = result.get("n_papers", 0)
            kappa = result.get("score_kappa", 0)
            wk = result.get("score_weighted_kappa", 0)
            corr = result.get("score_correlation", 0)
            overall = result.get("overall_agreement", 0)
            interp = result.get("interpretation", "?")
            self._log(
                f"Interrater complete: {n} papers, "
                f"κ={kappa:.3f}, wκ={wk:.3f}, "
                f"r={corr:.3f}, overall={overall:.3f} ({interp})")
            report_path = result.get("report_path", "")
            if report_path:
                self._log(f"  Report: {report_path}")
            try:
                kappa_val = float(kappa)
                summary = f"κ={kappa_val:.2f}, {n} papers"
            except (TypeError, ValueError):
                summary = f"{n} papers rated"
            self._set_pipeline_stage(8, "done", summary=summary)
        else:
            self._log(f"Interrater result: {result}")
            self._set_pipeline_stage(8, "done", summary="Interrater complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 8 interrater complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_interrater_error(self, err):
        self._log(f"INTERRATER ERROR: {err}", "ERROR")
        self._set_pipeline_stage(8, "failed", summary=self._pipeline_error_summary(err))
        self.interrater_btn.setEnabled(True)
        self.interrater_cancel_btn.setEnabled(False)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 8 (interrater error).", "ERROR")

    def _start_robustness(self):
        if not self._pipeline_stage_guard("Stage 9 (Robustness)"):
            return
        self._log("═" * 60)
        self._log("STAGE 9: ROBUSTNESS CHECKS")
        self._log("═" * 60)

        cfg = self.pipeline_config

        # Resolve corpus path — prefer custom path, then standard locations
        custom_input = self.robust_input.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            corpus_path = _Path(custom_input)
        else:
            filtered = cfg.claims_dir / "claims_filtered.json"
            base = cfg.claims_dir / "claims.json"
            relevant = cfg.corpus_dir / "relevant.json"
            if filtered.exists():
                corpus_path = filtered
            elif base.exists():
                corpus_path = base
            elif relevant.exists():
                corpus_path = relevant
            else:
                corpus_path = None

        if corpus_path is None or not corpus_path.exists():
            self._log(
                "No scored corpus found. Run Stage 2 (Score) or "
                "Stage 3 (Extract) first.", "ERROR")
            return

        import json as _json
        with open(corpus_path, encoding="utf-8") as f:
            papers = _json.load(f)
        if isinstance(papers, list):
            self._log(f"Input: {len(papers)} papers from {corpus_path}")
        else:
            self._log(f"Input: {corpus_path}")

        extra_kw: dict = {}
        if custom_input:
            extra_kw["claims_path"] = custom_input
        output_path = self.robust_output.text().strip()
        if output_path:
            extra_kw["output_dir"] = output_path

        self.robust_btn.setEnabled(False)
        self._active_stage_run_btn = self.robust_btn
        self._active_stage_cancel_btn = None

        self._pipeline_worker = PipelineStageWorker(
            stage_name="robustness",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(self._on_robustness_done, self._on_robustness_error)
        self._set_pipeline_stage(9, "running")
        self._pipeline_worker.start()

    def _on_robustness_done(self, result):
        self.robust_btn.setEnabled(True)
        if isinstance(result, dict):
            bs = result.get("bootstrap", {})
            sh = result.get("split_half", {})
            pt = result.get("permutation", {})
            sa = result.get("sensitivity", {})
            sens = result.get("condition_sensitivity", {})

            if bs:
                self._log(
                    f"Bootstrap CI: mean={bs.get('mean', 0):.3f}, "
                    f"95% CI=[{bs.get('ci_lower', 0):.3f}, "
                    f"{bs.get('ci_upper', 0):.3f}]")
            if sh:
                self._log(
                    f"Split-half: r={sh.get('mean_correlation', 0):.3f} "
                    f"({sh.get('interpretation', '?')})")
            if sa:
                self._log(
                    f"Sensitivity: stable={sa.get('stable', '?')}")
            if pt:
                self._log(
                    f"Permutation: r={pt.get('observed_correlation', 0):.4f}, "
                    f"p={pt.get('p_value', 1):.4f}, "
                    f"significant={pt.get('significant', False)}")
            if sens:
                from stages.s9_robustness import condition_sensitivity_summary
                self._log(condition_sensitivity_summary(sens))

            report = result.get("report_path", "")
            if report:
                self._log(f"  Report: {report}")

            # Auto-load network graph into the Neuroimaging tab
            net_path = result.get("results_path", "")
            if net_path:
                try:
                    import json as _nj
                    from pathlib import Path as _Path
                    from process.graph_analysis import network_artifact_path
                    net_json = network_artifact_path(_Path(net_path).parent)
                    if net_json.exists():
                        with open(net_json, encoding="utf-8") as f:
                            net_data = _nj.load(f)
                        # Convert Stage 9 edge list to widget schema
                        raw_edges = net_data.get("edges", [])
                        cond_weights: dict[str, int] = {}
                        reg_weights: dict[str, int] = {}
                        reg_types: dict[str, str] = {}
                        for e in raw_edges:
                            src = e.get("source", "")
                            tgt = e.get("target", "")
                            w = e.get("weight", 1)
                            if src:
                                cond_weights[src] = cond_weights.get(src, 0) + w
                            if tgt:
                                reg_weights[tgt] = reg_weights.get(tgt, 0) + w
                                if tgt not in reg_types:
                                    reg_types[tgt] = e.get("target_type", "entity")
                        widget_data = {
                            "condition_nodes": [
                                {"name": n, "paper_count": c}
                                for n, c in cond_weights.items()
                            ],
                            "region_nodes": [
                                {"name": n, "type": reg_types.get(n, "entity"),
                                 "paper_count": c}
                                for n, c in reg_weights.items()
                            ],
                            "condition_region_edges": [
                                {"condition": e["source"], "region": e["target"],
                                 "weight": e["weight"]}
                                for e in raw_edges
                                if e.get("source") and e.get("target")
                            ],
                        }
                        self.neuro_network_widget.set_network_data(widget_data)
                        self.neuro_export_graph_btn.setEnabled(True)
                        self._log(
                            f"Network graph auto-loaded: "
                            f"{net_data.get('n_edges', 0)} edges, "
                            f"{net_data.get('n_conditions', 0)} conditions")
                except Exception as exc:
                    logger.warning("Auto-load network graph failed: %s", exc)
        else:
            self._log(f"Robustness result: {result}")
        if isinstance(result, dict):
            n_papers = result.get("n_papers", 0)
            if n_papers:
                self._set_pipeline_stage(
                    9, "done",
                    summary=f"{n_papers} papers analysed",
                )
            else:
                self._set_pipeline_stage(9, "done", summary="Robustness complete")
        else:
            self._set_pipeline_stage(9, "done", summary="Robustness complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 9 robustness complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_robustness_error(self, err):
        self._log(f"ROBUSTNESS ERROR: {err}", "ERROR")
        self._set_pipeline_stage(9, "failed", summary=self._pipeline_error_summary(err))
        self.robust_btn.setEnabled(True)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 9 (robustness error).", "ERROR")

    def _start_verify(self):
        if not self._pipeline_stage_guard("Stage 10 (Verify)"):
            return
        self._log("═" * 60)
        self._log("STAGE 10: ABSTRACT VERIFICATION")
        self._log("═" * 60)

        cfg = self.pipeline_config

        # Resolve corpus path — prefer custom path, then standard locations
        custom_input = self.verify_claims.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            corpus_path = _Path(custom_input)
        else:
            filtered = cfg.claims_dir / "claims_filtered.json"
            base = cfg.claims_dir / "claims.json"
            relevant = cfg.corpus_dir / "relevant.json"
            if filtered.exists():
                corpus_path = filtered
            elif base.exists():
                corpus_path = base
            elif relevant.exists():
                corpus_path = relevant
            else:
                corpus_path = None

        if corpus_path is None or not corpus_path.exists():
            self._log(
                "No claims corpus found. Run Stage 3 (Extract) or "
                "Stage 5 (Audit) first.", "ERROR")
            return

        # Also check for abstract
        abs_path = cfg.clusters_dir / "abstract_draft.md"
        if not abs_path.exists():
            abs_path = cfg.output_dir / "abstract_draft.md"
        if not abs_path.exists():
            self._log(
                "No abstract_draft.md found. Run Stage 6 (Synthesize) first. "
                "Stage 10 will proceed but return empty results.", "WARNING")

        import json as _json
        with open(corpus_path, encoding="utf-8") as f:
            papers = _json.load(f)
        if isinstance(papers, list):
            self._log(f"Input: {len(papers)} papers from {corpus_path}")
        else:
            self._log(f"Input: {corpus_path}")

        extra_kw: dict = {}
        if custom_input:
            extra_kw["claims_path"] = custom_input

        self.verify_report_view.clear()
        self.verify_btn.setEnabled(False)
        self._active_stage_run_btn = self.verify_btn
        self._active_stage_cancel_btn = None

        self._pipeline_worker = PipelineStageWorker(
            stage_name="verify",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(self._on_verify_done, self._on_verify_error)
        self._set_pipeline_stage(10, "running")
        self._pipeline_worker.start()

    def _on_verify_done(self, result):
        self.verify_btn.setEnabled(True)
        if isinstance(result, dict):
            if result.get("error"):
                self._log(f"Verify error: {result['error']}", "ERROR")
                self._set_pipeline_stage(
                    10, "failed",
                    summary=self._pipeline_error_summary(result["error"]),
                )
            elif result.get("warning"):
                self._log(f"Verify warning: {result['warning']}", "WARNING")
            summary = result.get("summary", {})
            total = summary.get("total_checks", 0)
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            warnings = summary.get("warnings", 0)
            self._log(
                f"Verification complete: {total} checks — "
                f"{passed} passed, {failed} failed, {warnings} warnings")
            report = result.get("report_path", "")
            if report:
                self._log(f"  Report: {report}")
                try:
                    from pathlib import Path as _Path
                    report_text = _Path(report).read_text(encoding="utf-8")
                    self.verify_report_view.setMarkdown(report_text)
                except Exception as exc:
                    logger.warning("Could not load verification report for display: %s", exc)
            if not result.get("error"):
                self._set_pipeline_stage(
                    10, "done",
                    summary=f"{total} checks, {passed} passed",
                )
        else:
            self._log(f"Verify result: {result}")
            self._set_pipeline_stage(10, "done", summary="Verification complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 10 verify complete")
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL STAGES finished at stage 10 (verify).")

    def _on_verify_error(self, err):
        self._log(f"VERIFY ERROR: {err}", "ERROR")
        self._set_pipeline_stage(10, "failed", summary=self._pipeline_error_summary(err))
        self.verify_btn.setEnabled(True)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 10 (verify error).", "ERROR")

    def _start_meta_analysis(self):
        """Run meta-analytic synthesis (effect size pooling + bias tests)."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from process.meta_analysis import run_full_analysis

        self.meta_analysis_status.setText("Running meta-analysis...")
        self.meta_analysis_btn.setEnabled(False)
        self._active_stage_run_btn = self.meta_analysis_btn
        self._active_stage_cancel_btn = None
        QApplication.processEvents()

        try:
            results = run_full_analysis(self.pipeline_config, input_source="stage2c")

            if results.get("skipped"):
                reason = results.get("reason", "Unknown reason")
                self.meta_analysis_status.setText(f"Skipped: {reason}")
                self.meta_summary.setPlainText(f"Meta-analysis skipped.\n\n{reason}")
                self._log(f"Meta-analysis skipped: {reason}")
                return

            n = results.get("n_studies", 0)
            pooled = results.get("pooled")
            bias = results.get("bias")

            # Update forest plot canvas
            forest_fig = results.get("forest_fig")
            if forest_fig is not None:
                new_forest = FigureCanvasQTAgg(forest_fig)
                idx = self.meta_plot_tabs.indexOf(self.forest_canvas)
                self.meta_plot_tabs.removeTab(idx)
                self.meta_plot_tabs.insertTab(idx, new_forest, "Forest Plot")
                self.forest_canvas = new_forest

            # Update funnel plot canvas
            funnel_fig = results.get("funnel_fig")
            if funnel_fig is not None:
                new_funnel = FigureCanvasQTAgg(funnel_fig)
                idx = self.meta_plot_tabs.indexOf(self.funnel_canvas)
                self.meta_plot_tabs.removeTab(idx)
                self.meta_plot_tabs.insertTab(idx, new_funnel, "Funnel Plot")
                self.funnel_canvas = new_funnel

            # Build summary text
            lines = []
            lines.append(f"=== Meta-Analysis Summary ({n} studies) ===\n")

            if pooled:
                lines.append(f"Pooled Effect (RE):  {pooled.pooled_effect:.4f}")
                lines.append(f"95% CI:              [{pooled.ci_lower:.4f}, {pooled.ci_upper:.4f}]")
                lines.append(f"Z = {pooled.z_value:.3f}, p = {pooled.p_value:.4f}")
                lines.append(f"\nHeterogeneity:")
                lines.append(f"  I\u00b2 = {pooled.i_squared:.1f}%")
                lines.append(f"  \u03c4\u00b2 = {pooled.tau_squared:.6f}")
                lines.append(f"  Q  = {pooled.q_statistic:.3f} (p = {pooled.q_p_value:.4f})")
                if pooled.prediction_interval != (0.0, 0.0):
                    pi = pooled.prediction_interval
                    lines.append(f"  Prediction interval: [{pi[0]:.4f}, {pi[1]:.4f}]")

            if bias:
                lines.append(f"\nPublication Bias Tests:")
                lines.append(f"  Egger's intercept = {bias.eggers_intercept:.4f} "
                             f"(t = {bias.eggers_t:.3f}, p = {bias.eggers_p:.4f})")
                lines.append(f"  {bias.eggers_interpretation}")
                lines.append(f"\n  Trim-and-fill: {bias.trimfill_n_imputed} studies imputed")
                if bias.trimfill_n_imputed > 0:
                    lines.append(f"  Adjusted effect: {bias.trimfill_adjusted_effect:.4f} "
                                 f"[{bias.trimfill_adjusted_ci_lower:.4f}, "
                                 f"{bias.trimfill_adjusted_ci_upper:.4f}]")
                lines.append(f"\n  Fail-safe N = {bias.failsafe_n}")
                lines.append(f"  {bias.failsafe_interpretation}")
                lines.append(f"\n  Overall: {bias.overall_interpretation}")

            if n < 10:
                lines.append(
                    f"\n\u26a0 Note: Publication bias tests require \u2265 10 studies "
                    f"for adequate power. Current analysis has {n} studies."
                )

            self.meta_summary.setPlainText("\n".join(lines))
            self.meta_analysis_status.setText(
                f"Complete \u2014 {n} studies pooled."
            )
            self._log(f"Meta-analysis complete: {n} studies, "
                      f"pooled effect = {pooled.pooled_effect:.4f}" if pooled else "")

        except Exception as exc:
            self.meta_analysis_status.setText(f"Error: {exc}")
            self.meta_summary.setPlainText(f"Meta-analysis failed:\n\n{exc}")
            self._log(f"Meta-analysis error: {exc}")
            logger.exception("Meta-analysis failed")
        finally:
            self.meta_analysis_btn.setEnabled(True)

    def _start_compare_runs(self):
        self._log("=" * 60)
        self._log("STAGE 11: COMPARE RUNS")
        self._log("=" * 60)

        from pathlib import Path as _Path

        path_a = self.compare_run_a.text().strip()
        path_b = self.compare_run_b.text().strip()

        if not path_a or not path_b:
            self._log("Please specify both Run A and Run B file paths.", "ERROR")
            return

        path_a = _Path(path_a)
        path_b = _Path(path_b)

        if not path_a.exists():
            self._log(f"Run A file not found: {path_a}", "ERROR")
            return
        if not path_b.exists():
            self._log(f"Run B file not found: {path_b}", "ERROR")
            return

        self._log(f"Run A: {path_a}")
        self._log(f"Run B: {path_b}")

        comparison_name = "run_compare"
        output_path = self.compare_output.text().strip()
        if output_path:
            comparison_name = _Path(output_path).stem

        self.compare_btn.setEnabled(False)
        self.compare_progress.setVisible(True)
        self.compare_progress.setRange(0, 0)  # indeterminate
        self.compare_status.setText("Comparing runs...")

        self._pipeline_worker = PipelineStageWorker(
            stage_name="compare_runs",
            config=self.pipeline_config,
            claims_paths=[str(path_a), str(path_b)],
            comparison_name=comparison_name,
        )
        self._connect_stage_worker(self._on_compare_done, self._on_compare_error)
        self._set_pipeline_stage(11, "running")
        self._pipeline_worker.start()

    def _on_compare_done(self, result):
        self.compare_btn.setEnabled(True)
        self.compare_progress.setVisible(False)
        self.compare_status.setText("Comparison complete.")
        if isinstance(result, dict):
            overall = result.get("overall_summary", {})
            stability = overall.get("stability_score")
            interp = overall.get("interpretation", "N/A")
            n_runs = result.get("n_runs", 0)
            pairwise = result.get("pairwise_comparisons", [])

            self._log(f"Compared {n_runs} runs, {len(pairwise)} pair(s)")
            if stability is not None:
                self._log(f"Stability: {stability:.3f} ({interp})")
            else:
                self._log(f"Stability: {interp}")

            for i, p in enumerate(pairwise, 1):
                shared = p.get("shared_papers", 0)
                pj = p.get("paper_jaccard")
                wk = p.get("score_weighted_kappa")
                self._log(
                    f"  Pair {i}: {shared} shared papers, "
                    f"Jaccard={pj:.3f if pj is not None else 'N/A'}, "
                    f"wKappa={wk:.3f if wk is not None else 'N/A'}")

            report_path = result.get("report_path", "")
            if report_path:
                self._log(f"Report: {report_path}")
                try:
                    with open(report_path, encoding="utf-8") as f:
                        self.compare_results.setPlainText(f.read())
                except OSError:
                    pass
            summary = f"{n_runs} runs compared"
            if interp and interp != "N/A":
                summary = f"{summary} ({interp})"
            self._set_pipeline_stage(11, "done", summary=summary)
        else:
            self._log(f"Compare result: {result}")
            self._set_pipeline_stage(11, "done", summary="Comparison complete")
        self._log("=" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 11 compare complete")

    def _on_compare_error(self, err):
        self._log(f"COMPARE RUNS ERROR: {err}", "ERROR")
        self._set_pipeline_stage(11, "failed", summary=self._pipeline_error_summary(err))
        self.compare_btn.setEnabled(True)
        self.compare_progress.setVisible(False)
        self.compare_status.setText(f"Error: {err}")
