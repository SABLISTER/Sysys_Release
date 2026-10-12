import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from core.config import Config
from gui.dialogs import SettingsDialog
from gui.main_window import MainWindow
from gui.original_question import OriginalQuestionState


def _app() -> QApplication:
    return cast(QApplication, QApplication.instance() or QApplication([]))


def test_first_search_freezes_original_and_later_queries_diverge():
    state = OriginalQuestionState()
    state.set_prospective("Original exact question?")

    assert state.capture_first_search("Original exact question?") == (
        "Original exact question?"
    )
    assert state.captured is True

    state.set_prospective("Expanded search query")
    assert state.value == "Original exact question?"
    assert state.question_for_model("Expanded search query") == (
        "Original exact question?"
    )


def test_existing_project_migrates_saved_search_query():
    state = OriginalQuestionState.from_window_state(
        {"search_query": "Saved legacy question?"}
    )

    assert state.value == "Saved legacy question?"
    assert state.captured is True


def test_explicit_edit_changes_only_original():
    state = OriginalQuestionState("Original?", captured=True)

    state.edit_captured("Corrected original?")

    assert state.question_for_model("Evolving query") == "Corrected original?"


def test_from_window_state_handles_none():
    state = OriginalQuestionState.from_window_state(None)

    assert state.value == ""
    assert state.captured is False


def test_from_window_state_handles_non_dict():
    state = OriginalQuestionState.from_window_state(["not", "a", "dict"])

    assert state.value == ""
    assert state.captured is False


def test_from_window_state_handles_malformed_string_input():
    state = OriginalQuestionState.from_window_state("not even a container")

    assert state.value == ""
    assert state.captured is False


def test_stage_config_uses_original_not_evolving_query(tmp_path):
    app = _app()
    provider = QComboBox()
    provider.addItem("lmstudio")
    provider.setCurrentText("lmstudio")
    model = QLineEdit("local-model")
    window: Any = SimpleNamespace(
        pipeline_config=Config(project_root=tmp_path),
        sysys_nl_query=QLineEdit("Expanded follow-up query"),
        _original_question_state=OriginalQuestionState(
            "Exact original question?", captured=True
        ),
    )

    cfg = MainWindow._apply_config_overrides_for_stage(window, provider, model)

    assert cfg.get_natural_language_question() == "Exact original question?"
    assert window.sysys_nl_query.text() == "Expanded follow-up query"
    assert app is not None


def test_project_window_state_preserves_both_questions():
    app = _app()
    window: Any = SimpleNamespace(
        sysys_nl_query=QLineEdit("Expanded follow-up query"),
        _original_question_state=OriginalQuestionState(
            "Exact original question?", captured=True
        ),
    )

    state = MainWindow._project_question_window_state(window)

    assert state == {
        "search_query": "Expanded follow-up query",
        "original_research_question": "Exact original question?",
        "original_question_captured": True,
    }
    assert app is not None


def test_apply_config_overrides_does_not_leak_original_into_hypothesis_text(tmp_path):
    app = _app()
    provider = QComboBox()
    provider.addItem("lmstudio")
    provider.setCurrentText("lmstudio")
    model = QLineEdit("local-model")
    pipeline_config = Config(project_root=tmp_path)
    pipeline_config.hypothesis_text = "Pre-existing hypothesis text"
    window: Any = SimpleNamespace(
        pipeline_config=pipeline_config,
        sysys_nl_query=QLineEdit("Expanded follow-up query"),
        _original_question_state=OriginalQuestionState(
            "Exact original question?", captured=True
        ),
    )

    cfg = MainWindow._apply_config_overrides_for_stage(window, provider, model)

    assert cfg.hypothesis_text == "Pre-existing hypothesis text"
    assert cfg.get_natural_language_question() == "Exact original question?"
    assert app is not None


def test_original_question_for_model_captures_and_saves_once_on_first_use(tmp_path):
    """Model-backed stages without a Stage 1 search must freeze the fallback
    question on first use so later stages cannot drift to a new value."""
    save_calls = []
    window: Any = SimpleNamespace(
        sysys_nl_query=QLineEdit("Fallback model question?"),
        _original_question_state=OriginalQuestionState(),
        _project_path=Path("/tmp/some-project"),
        _save_project=lambda: save_calls.append(1),
    )
    window._capture_original_question_for_first_search = (
        MainWindow._capture_original_question_for_first_search.__get__(window)
    )

    first = MainWindow._original_question_for_model(window)
    second = MainWindow._original_question_for_model(window)

    assert first == "Fallback model question?"
    assert second == "Fallback model question?"
    assert window._original_question_state.captured is True
    assert window._original_question_state.value == "Fallback model question?"
    assert save_calls == [1]


def test_assigning_first_project_path_preserves_captured_original(tmp_path):
    window: Any = SimpleNamespace(
        _project_path=None,
        _original_question_state=OriginalQuestionState(
            "Exact first search question?", captured=True
        ),
        pipeline_config=Config(project_root=tmp_path),
        setWindowTitle=lambda _title: None,
    )

    MainWindow._set_project_path(window, tmp_path / "first-project")

    assert window._project_path == Path(tmp_path / "first-project")
    assert window._original_question_state == OriginalQuestionState(
        "Exact first search question?", captured=True
    )


def test_new_project_from_open_project_resets_original_question(tmp_path):
    window: Any = SimpleNamespace(
        _project_path=tmp_path / "existing-project",
        _original_question_state=OriginalQuestionState(
            "Existing project original?", captured=True
        ),
    )

    MainWindow._prepare_for_new_project(window)

    assert window._original_question_state == OriginalQuestionState()


def test_preferences_locks_captured_original_until_explicit_edit(tmp_path):
    _app()
    window: Any = QWidget()
    window.pipeline_config = Config(project_root=tmp_path)
    window._project_path = tmp_path
    window._original_question_state = OriginalQuestionState(
        "Exact original?", captured=True
    )
    window._edit_original_research_question = (
        window._original_question_state.edit_captured
    )

    dialog = SettingsDialog(window)

    assert dialog.settings_original_question.text() == "Exact original?"
    assert dialog.settings_original_question.isReadOnly() is True

    dialog.settings_edit_original_question.setChecked(True)
    assert dialog.settings_original_question.isReadOnly() is False


def test_preferences_edit_updates_only_captured_original(tmp_path):
    _app()
    window: Any = QWidget()
    window.pipeline_config = Config(project_root=tmp_path)
    window._project_path = tmp_path
    window.sysys_nl_query = QLineEdit("Expanded search question")
    window._original_question_state = OriginalQuestionState(
        "Exact original?", captured=True
    )
    window._edit_original_research_question = (
        window._original_question_state.edit_captured
    )

    dialog = SettingsDialog(window)
    dialog.settings_edit_original_question.setChecked(True)
    dialog.settings_original_question.setText("Corrected original?")
    dialog._apply_original_question()

    assert window._original_question_state.value == "Corrected original?"
    assert window.sysys_nl_query.text() == "Expanded search question"


def test_pre_search_preferences_update_search_box(tmp_path):
    _app()
    window: Any = QWidget()
    window.pipeline_config = Config(project_root=tmp_path)
    window._project_path = tmp_path
    window.sysys_nl_query = QLineEdit()
    window._original_question_state = OriginalQuestionState()
    window._save_project = lambda: None

    dialog = SettingsDialog(window)
    dialog.settings_original_question.setText("Prospective original?")
    dialog._apply_original_question()

    assert window.sysys_nl_query.text() == "Prospective original?"
    assert window._original_question_state.captured is False


def test_preferences_disables_original_question_without_project(tmp_path):
    _app()
    window: Any = QWidget()
    window.pipeline_config = Config(project_root=tmp_path)
    window._project_path = None
    window.sysys_nl_query = QLineEdit()
    window._original_question_state = OriginalQuestionState()

    dialog = SettingsDialog(window)

    assert dialog.original_question_group.isEnabled() is False


def test_repeated_apply_relocks_edit_gate_after_successful_edit(tmp_path):
    """A second Apply click without re-checking the edit box must not
    re-save an accidental extra edit (Minor finding #7)."""
    _app()
    window: Any = QWidget()
    window.pipeline_config = Config(project_root=tmp_path)
    window._project_path = tmp_path
    window.sysys_nl_query = QLineEdit("Expanded search question")
    window._original_question_state = OriginalQuestionState(
        "Exact original?", captured=True
    )
    window._edit_original_research_question = (
        window._original_question_state.edit_captured
    )

    dialog = SettingsDialog(window)
    dialog.settings_edit_original_question.setChecked(True)
    dialog.settings_original_question.setText("Corrected original?")
    dialog._apply_original_question()

    assert window._original_question_state.value == "Corrected original?"
    assert dialog.settings_edit_original_question.isChecked() is False
    assert dialog.settings_original_question.isReadOnly() is True

    # Second Apply without re-checking "Edit" must be inert.
    dialog.settings_original_question.setText("Accidental extra edit?")
    dialog._apply_original_question()

    assert window._original_question_state.value == "Corrected original?"


def test_prepare_for_new_project_is_noop_without_existing_project():
    """Minor finding #8: the no-project branch of `_prepare_for_new_project`
    must leave any in-memory state untouched (there is nothing to reset)."""
    window: Any = SimpleNamespace(
        _project_path=None,
        _original_question_state=OriginalQuestionState(
            "Should remain untouched?", captured=True
        ),
    )

    MainWindow._prepare_for_new_project(window)

    assert window._original_question_state == OriginalQuestionState(
        "Should remain untouched?", captured=True
    )


# ── Finding #1 (CRITICAL): Stage 8 must use the captured/overridden config ──


class _FakeSignal:
    def connect(self, *_args, **_kwargs):
        pass


class _FakeStageWorker:
    """Records the kwargs it was constructed with instead of running a stage."""

    instances: list["_FakeStageWorker"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.progress = _FakeSignal()
        self.finished = _FakeSignal()
        self.error = _FakeSignal()
        self.cancelled = _FakeSignal()
        self.started = False
        _FakeStageWorker.instances.append(self)

    def start(self):
        self.started = True


def _interrater_window(tmp_path: Path) -> Any:
    pipeline_config = Config(project_root=tmp_path)
    pipeline_config.claims_dir.mkdir(parents=True, exist_ok=True)
    pipeline_config.corpus_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_config.claims_dir / "claims_filtered.json").write_text(
        json.dumps([{"title": "Paper A"}, {"title": "Paper B"}]),
        encoding="utf-8",
    )

    interrater_provider = QComboBox()
    interrater_provider.addItems(["lmstudio", "openai"])
    interrater_provider.setCurrentText("lmstudio")
    interrater_model = QLineEdit("second-rater-model")

    # Stage 2's widgets identify the PRIMARY rater — the one whose scores are
    # already in the corpus that Stage 8 compares against.
    score_provider = QComboBox()
    score_provider.addItems(["openai", "lmstudio"])
    score_provider.setCurrentText("openai")
    score_model = QLineEdit("primary-rater-model")

    window: Any = SimpleNamespace(
        pipeline_config=pipeline_config,
        sysys_nl_query=QLineEdit("Expanded follow-up query"),
        _original_question_state=OriginalQuestionState(
            "Exact original question?", captured=True
        ),
        _pipeline_worker=None,
        _search_worker=None,
        _pipeline_stage_guard=lambda _label: True,
        _log=lambda *_a, **_k: None,
        interrater_provider=interrater_provider,
        interrater_model=interrater_model,
        score_provider=score_provider,
        score_model=score_model,
        interrater_corpus=QLineEdit(""),
        interrater_sample=QSpinBox(),
        interrater_btn=QPushButton(),
        interrater_cancel_btn=QPushButton(),
        _active_stage_run_btn=None,
        _active_stage_cancel_btn=None,
        _on_interrater_done=lambda _result: None,
        _on_interrater_error=lambda _err: None,
        _on_interrater_cancelled=lambda: None,
        _set_pipeline_stage=lambda *_a, **_k: None,
    )
    window.interrater_sample.setRange(1, 500)
    window.interrater_sample.setValue(10)
    window._apply_config_overrides_for_stage = (
        MainWindow._apply_config_overrides_for_stage.__get__(window)
    )
    window._original_question_for_model = (
        MainWindow._original_question_for_model.__get__(window)
    )
    window._capture_original_question_for_first_search = (
        MainWindow._capture_original_question_for_first_search.__get__(window)
    )
    window._connect_stage_worker = (
        MainWindow._connect_stage_worker.__get__(window)
    )
    window._project_path = tmp_path
    window._save_project = lambda: None
    return window


def test_start_interrater_passes_overridden_config_not_raw_pipeline_config(
    tmp_path, monkeypatch
):
    _app()
    _FakeStageWorker.instances.clear()
    monkeypatch.setattr("gui.main_window.PipelineStageWorker", _FakeStageWorker)

    window = _interrater_window(tmp_path)

    MainWindow._start_interrater(window)

    assert len(_FakeStageWorker.instances) == 1
    kwargs = _FakeStageWorker.instances[0].kwargs
    used_config = kwargs["config"]
    assert used_config is not window.pipeline_config
    assert used_config.get_natural_language_question() == (
        "Exact original question?"
    )

    # The config must describe the PRIMARY rater. Building it from the
    # Interrater tab instead would set llm_provider to the second rater, so
    # Stage 8 could not tell the two apart and every run would look like a
    # matched pair rating itself.
    assert used_config.llm_provider == "openai"
    assert used_config.openai_model == "primary-rater-model"
    assert kwargs["primary_provider"] == "openai"
    assert kwargs["primary_model"] == "primary-rater-model"

    # The second rater travels as its own arguments.
    assert kwargs["second_provider"] == "lmstudio"
    assert kwargs["second_model"] == "second-rater-model"


# ── Finding #4: first-search capture wiring and open/restore migration ──


class _RecordingSearchWorker:
    instances: list["_RecordingSearchWorker"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.progress = _FakeSignal()
        self.finished_search = _FakeSignal()
        self.error = _FakeSignal()
        self.started = False
        _RecordingSearchWorker.instances.append(self)

    def start(self):
        self.started = True


class _DummySearchWindow:
    """Minimal double exercising the real capture/sync/save wiring used by
    `_start_sysys_search`, without instantiating the full MainWindow UI."""

    def __init__(self, tmp_path: Path, query: str):
        self.pipeline_config = Config(project_root=tmp_path)
        self.sysys_nl_query = QLineEdit(query)
        self._original_question_state = OriginalQuestionState()
        self._project_path = tmp_path
        self._pipeline_worker = None
        self._search_worker = None
        self.stat_extract_llm_context = QCheckBox()
        self.hypothesis_components = []
        self.src_pubmed = QCheckBox()
        self.src_pubmed.setChecked(True)
        self.src_s2 = QCheckBox()
        self.src_openalex = QCheckBox()
        self.src_elicit = QCheckBox()
        self.src_scopus = QCheckBox()
        self.src_wos = QCheckBox()
        self.paper_limit = QSpinBox()
        self.paper_limit.setRange(1, 1000)
        self.paper_limit.setValue(50)
        self.sysys_search_btn = QPushButton()
        self.sysys_cancel_btn = QPushButton()
        self.search_progress = _FakeProgressBar()
        self.pipeline_state = None
        self.logs: list[str] = []
        self.save_calls: list[int] = []
        self.save_snapshots: list[int] = []
        self.stage_updates: list[tuple[int, str]] = []

        self._pipeline_stage_guard = MainWindow._pipeline_stage_guard.__get__(self)
        self._capture_original_question_for_first_search = (
            MainWindow._capture_original_question_for_first_search.__get__(self)
        )
        self._sync_query_components_from_question = (
            MainWindow._sync_query_components_from_question.__get__(self)
        )
        self._start_sysys_search = MainWindow._start_sysys_search.__get__(self)

    def _log(self, msg, level="INFO"):
        self.logs.append(msg)

    def _refresh_component_list(self):
        pass

    def _update_search_workflow_buttons(self):
        pass

    def _save_project(self):
        self.save_calls.append(1)
        self.save_snapshots.append(len(self.hypothesis_components))

    def _on_sysys_search_progress(self, _msg):
        pass

    def _on_sysys_search_done(self, _corpus):
        pass

    def _on_sysys_search_error(self, _err):
        pass

    def _set_pipeline_stage(self, stage, status):
        self.stage_updates.append((stage, status))


class _FakeProgressBar:
    def setVisible(self, _visible):
        pass

    def setRange(self, _lo, _hi):
        pass

    def setValue(self, _value):
        pass


def test_start_sysys_search_captures_original_question_and_saves_once(
    tmp_path, monkeypatch
):
    _app()
    _RecordingSearchWorker.instances.clear()
    monkeypatch.setattr(
        "gui.main_window.PipelineSearchWorker", _RecordingSearchWorker
    )

    window = _DummySearchWindow(
        tmp_path, "Does meditation reduce anxiety in college students?"
    )

    window._start_sysys_search()

    assert window._original_question_state.captured is True
    assert window._original_question_state.value == (
        "Does meditation reduce anxiety in college students?"
    )
    assert len(_RecordingSearchWorker.instances) == 1
    assert _RecordingSearchWorker.instances[0].kwargs[
        "natural_language_question"
    ] == "Does meditation reduce anxiety in college students?"
    assert _RecordingSearchWorker.instances[0].started is True
    assert window.save_calls == [1]
    assert window.stage_updates == [(1, "running")]


def test_start_sysys_search_saves_after_syncing_components_not_stale(
    tmp_path, monkeypatch
):
    """Minor finding #9: the project save triggered by first-search capture
    must reflect post-sync hypothesis components, not the pre-sync (stale,
    empty) list."""
    _app()
    _RecordingSearchWorker.instances.clear()
    monkeypatch.setattr(
        "gui.main_window.PipelineSearchWorker", _RecordingSearchWorker
    )

    window = _DummySearchWindow(
        tmp_path, "Does meditation reduce anxiety in college students?"
    )

    window._start_sysys_search()

    assert window.save_snapshots == [len(window.hypothesis_components)]
    assert window.save_snapshots[0] > 0


def test_open_project_migrates_legacy_search_query(tmp_path):
    _app()
    project_path = tmp_path / "legacy-project"
    project_path.mkdir()
    (project_path / "project_state.json").write_text(
        json.dumps({"window_state": {"search_query": "Legacy saved question?"}}),
        encoding="utf-8",
    )

    window: Any = SimpleNamespace(
        _original_question_state=OriginalQuestionState(),
        _project_path=None,
        pipeline_config=Config(project_root=tmp_path),
        setWindowTitle=lambda *_a: None,
        _populate_recent_menu=lambda: None,
        sysys_nl_query=QLineEdit(),
        statusBar=lambda: SimpleNamespace(showMessage=lambda *a, **k: None),
    )
    window._set_project_path = MainWindow._set_project_path.__get__(window)
    window._restore_fulltext_project_config = MainWindow._restore_fulltext_project_config.__get__(window)
    window._refresh_pipeline_progress_from_disk = lambda cfg: None

    MainWindow._open_project_path(window, project_path)

    assert window._original_question_state.value == "Legacy saved question?"
    assert window._original_question_state.captured is True
    assert window.sysys_nl_query.text() == "Legacy saved question?"


def test_open_project_restores_original_before_setting_gui_field(tmp_path):
    """Regression for finding #4: when only the new
    `original_research_question` field is saved (no `search_query`), the GUI
    field must be populated from the *restored* state. If `_open_project_path`
    stopped restoring the state before computing the GUI text, this value
    would stay empty."""
    _app()
    project_path = tmp_path / "captured-project"
    project_path.mkdir()
    (project_path / "project_state.json").write_text(
        json.dumps(
            {
                "window_state": {
                    "original_research_question": "Captured original question?",
                    "original_question_captured": True,
                }
            }
        ),
        encoding="utf-8",
    )

    window: Any = SimpleNamespace(
        _original_question_state=OriginalQuestionState(),
        _project_path=None,
        pipeline_config=Config(project_root=tmp_path),
        setWindowTitle=lambda *_a: None,
        _populate_recent_menu=lambda: None,
        sysys_nl_query=QLineEdit(),
        statusBar=lambda: SimpleNamespace(showMessage=lambda *a, **k: None),
    )
    window._set_project_path = MainWindow._set_project_path.__get__(window)
    window._restore_fulltext_project_config = MainWindow._restore_fulltext_project_config.__get__(window)
    window._refresh_pipeline_progress_from_disk = lambda cfg: None

    MainWindow._open_project_path(window, project_path)

    assert window._original_question_state.value == "Captured original question?"
    assert window._original_question_state.captured is True
    assert window.sysys_nl_query.text() == "Captured original question?"
