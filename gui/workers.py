"""
Worker threads for the Hypothesis Consensus Analyzer GUI.

Each worker runs a pipeline stage or analysis task in a background QThread,
emitting progress/status/finished/error signals back to the main thread.
"""
import logging
import threading
import traceback
from concurrent.futures import CancelledError as FutureCancelledError
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from process.text_scaling import TextScalingCancelled, run_text_scaling

logger = logging.getLogger(__name__)


def _progress_message(payload: str | dict[str, object] | None) -> str:
    """Flatten a stage progress report into text for a Qt ``Signal(str)``.

    Stages report progress either as a plain string or as a structured dict
    (``{"message", "current", "total"}``). Qt signals are statically typed, so
    the dict form has to be flattened before it crosses the thread boundary:
    emitting a dict on a ``Signal(str)`` is dropped by Shiboken with only a
    ``_pythonToCppCopy`` warning on stderr, so the update is lost silently.
    """
    if isinstance(payload, dict):
        message = payload.get("message")
        return "" if message is None else str(message)
    return "" if payload is None else str(payload)


def _progress_count_values(
    payload: str | dict[str, object] | None,
) -> tuple[int, int] | None:
    """Return validated ``(current, total)`` values from structured progress."""
    if not isinstance(payload, dict):
        return None
    current = payload.get("current")
    total = payload.get("total")
    if (
        not isinstance(current, int)
        or isinstance(current, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
    ):
        return None
    return current, total


def _fulltext_result_success(result: dict) -> bool:
    """Return whether a full-text retrieval result contains usable text."""
    if bool(result.get("success")):
        return True
    text = result.get("text") or result.get("full_text") or result.get("fulltext")
    return isinstance(text, str) and bool(text.strip())


class AnalysisWorker(QThread):
    progress = Signal(int, int)
    status = Signal(str)
    finished_analysis = Signal(object)
    error = Signal(str)

    def __init__(self, articles=None, components=None, **kwargs):
        super().__init__()
        self.articles = articles or {}
        self.components = components or []
        self._kwargs = kwargs

    def run(self):
        try:
            from analysis.scoring import run_full_analysis

            logger.info("AnalysisWorker: %d articles, %d components",
                        len(self.articles), len(self.components))

            def progress_cb(current, total):
                self.progress.emit(current, total)

            session = run_full_analysis(
                self.articles, self.components, progress_cb=progress_cb
            )
            self.finished_analysis.emit(session)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("AnalysisWorker ERROR: %s\n%s", exc, tb)
            self.error.emit(str(exc))


class FullTextRetrievalWorker(QThread):
    """v3 full-text retrieval worker."""
    progress = Signal(int, int)
    status = Signal(str)
    finished_retrieval = Signal(int, int, str)
    error = Signal(str)

    def __init__(self, articles=None, output_path="", config=None, **kwargs):
        super().__init__()
        self.articles = articles or {}
        self.output_path = output_path
        self.config = config
        self._kwargs = kwargs
        self._cancel = False
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel = True
        self._cancel_event.set()

    def run(self):
        try:
            from pathlib import Path
            from stages.s7_fulltext import check_oa_status, retrieve_fulltext

            logger.info("FullTextRetrievalWorker: %d articles", len(self.articles))

            # Build targets from articles with DOIs
            targets = []
            for art_id, art in self.articles.items():
                doi = art.doi if hasattr(art, 'doi') else art.get('doi', '')
                if doi:
                    targets.append({"doi": doi, "title": getattr(art, 'title', ''),
                                    "article_id": art_id})

            if not targets:
                self.finished_retrieval.emit(0, 0, "")
                return

            self.status.emit(f"Checking OA for {len(targets)} papers...")
            email = "research@pipeline.dev"
            cache_path = None
            if self.config:
                email = getattr(self.config, 'pubmed_email', '') or email
                # Share the persistent Unpaywall cache with pipeline Stage 7
                # so corpus checks are only paid for once across runs.
                data_dir = getattr(self.config, 'data_dir', None)
                if data_dir:
                    cache_dir = Path(data_dir) / "fulltext"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path = cache_dir / "oa_status.json"

            oa_papers = check_oa_status(
                targets, email=email,
                progress_callback=self.status.emit,
                cache_path=cache_path,
                cancel_event=self._cancel_event,
            )
            if self._cancel_event.is_set():
                self.status.emit("Full-text retrieval cancelled during OA check.")
                self.finished_retrieval.emit(0, len(targets), "")
                return

            oa_avail = [p for p in oa_papers if p.get('is_oa')]

            if not oa_avail:
                self.finished_retrieval.emit(0, len(targets), "")
                return

            import tempfile
            out_dir = self.output_path or tempfile.mkdtemp(prefix="sysys_ft_")
            results = retrieve_fulltext(
                oa_avail, Path(out_dir),
                progress_callback=self.status.emit,
                cancel_event=self._cancel_event,
            )
            success = sum(1 for r in results if _fulltext_result_success(r))

            self.finished_retrieval.emit(success, len(targets), str(out_dir))
        except Exception as exc:
            tb_str = traceback.format_exc()
            logger.error("FullTextRetrievalWorker ERROR: %s\n%s", exc, tb_str)
            self.error.emit(str(exc))


class TextScalingWorker(QThread):
    """Text scaling (Wordfish) and stylometric profiling worker."""
    progress = Signal(int, int)
    status = Signal(str)
    finished_scaling = Signal(object, dict, dict)
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, articles=None, components=None,
                 min_doc_words=50, max_features=5000, **kwargs):
        super().__init__()
        self.articles = articles or {}
        self.components = components or []
        self.min_doc_words = min_doc_words
        self.max_features = max_features
        self._kwargs = kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            logger.info("TextScalingWorker: %d articles, min_words=%d, max_features=%d",
                        len(self.articles), self.min_doc_words, self.max_features)

            model, profiles, statistics = run_text_scaling(
                self.articles,
                min_doc_words=self.min_doc_words,
                max_features=self.max_features,
                progress_cb=self.progress.emit,
                status_cb=self.status.emit,
                cancel_cb=lambda: self._cancel,
            )
            self.finished_scaling.emit(model, profiles, statistics)
        except TextScalingCancelled:
            logger.info("TextScalingWorker: cancelled")
            self.cancelled.emit()
        except Exception as exc:
            tb_str = traceback.format_exc()
            logger.error("TextScalingWorker ERROR: %s\n%s", exc, tb_str)
            self.error.emit(str(exc))


class ProjectBackupWorker(QThread):
    """Incrementally backs up a project folder without blocking the GUI."""

    progress = Signal(str)
    finished_backup = Signal(dict)
    error = Signal(str)

    def __init__(self, project_path: Path, destination_dir: Path, **options):
        super().__init__()
        self.project_path = Path(project_path)
        self.destination_dir = Path(destination_dir)
        self.options = options

    def run(self):
        try:
            from gui.project_manager import ProjectManager

            self.progress.emit("Starting project auto-backup...")
            result = ProjectManager.sync_project_backup(
                self.project_path,
                self.destination_dir,
                **self.options,
            )
            self.progress.emit("Project auto-backup complete.")
            self.finished_backup.emit(result)
        except Exception as exc:
            tb_str = traceback.format_exc()
            logger.error("ProjectBackupWorker ERROR: %s\n%s", exc, tb_str)
            self.error.emit(str(exc))


class PipelineSearchWorker(QThread):
    """Runs Stage 1 (literature search) in a background thread.

    Creates an AsyncBridge to run the async search_runner coroutines,
    emitting progress updates and the final corpus list.
    """

    progress = Signal(str)
    status = Signal(str)
    finished_search = Signal(list)
    cancelled = Signal()
    error = Signal(str)

    def __init__(
        self,
        natural_language_question: str = "",
        semantic_terms=None,
        boolean_terms=None,
        provider=None,
        config=None,
        max_results: int = 500,
        **kwargs,
    ):
        super().__init__()
        self.nl_question = natural_language_question
        self.semantic_terms = semantic_terms or []
        self.boolean_terms = boolean_terms or []
        self.provider = provider
        self.config = config
        self.max_results = max_results
        self._kwargs = kwargs
        self._cancel = False
        self._cancel_event = threading.Event()
        self._bridge = None
        self._future = None

    def cancel(self):
        self._cancel = True
        self._cancel_event.set()
        future = self._future
        if future and not future.done():
            future.cancel()

    def run(self):
        logger.info("═" * 60)
        logger.info("PipelineSearchWorker.run() STARTED")
        logger.info("═" * 60)
        logger.info("  NL question: %s", self.nl_question)
        logger.info("  Semantic terms: %s", self.semantic_terms)
        logger.info("  Boolean terms: %s", self.boolean_terms)
        logger.info("  Provider: %s", self.provider)
        logger.info("  Paper limit: %d total across selected providers", self.max_results)

        try:
            from gui.search_runner import AsyncBridge, ProviderChoice, run_gui_search

            bridge = AsyncBridge()
            self._bridge = bridge

            def progress_cb(msg: str) -> None:
                logger.info("Search progress: %s", msg)
                self.progress.emit(msg)

            if self._cancel_event.is_set():
                logger.info("PipelineSearchWorker: cancelled before submit")
                self.cancelled.emit()
                return

            self.progress.emit("Initializing search...")

            if isinstance(self.provider, list):
                provider = self.provider
            else:
                provider = self.provider or ProviderChoice.ALL
            logger.info("Running search with provider=%s", provider)

            try:
                future = bridge.submit(
                    run_gui_search(
                        natural_language_question=self.nl_question,
                        semantic_terms=self.semantic_terms,
                        boolean_terms=self.boolean_terms,
                        provider=provider,
                        config=self.config,
                        max_results=self.max_results,
                        progress_callback=progress_cb,
                        cancel_event=self._cancel_event,
                    )
                )
                self._future = future

                # Block until the coroutine completes
                result = future.result()
            finally:
                bridge.shutdown()
                self._bridge = None
                self._future = None

            logger.info("PipelineSearchWorker: search returned %d papers", len(result))
            self.progress.emit(f"Search complete: {len(result)} unique papers")
            self.finished_search.emit(result)

        except FutureCancelledError:
            logger.info("PipelineSearchWorker cancelled")
            self.cancelled.emit()
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("PipelineSearchWorker ERROR: %s\n%s", exc, tb)
            self.error.emit(f"Search failed: {exc}")


class PipelineStageWorker(QThread):
    """Runs any pipeline stage (2-11) in a background thread.

    Looks up the stage in _STAGE_REGISTRY and runs it via AsyncBridge.
    """

    progress = Signal(str)
    progress_counts = Signal(int, int)
    finished = Signal(str, object)
    error = Signal(str, str)
    cancelled = Signal()

    def __init__(self, stage_name: str = "", config=None, **kwargs):
        super().__init__()
        self.stage_name = stage_name
        self.config = config
        self._kwargs = kwargs
        self._cancel = False
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel = True
        self._cancel_event.set()

    def run(self):
        logger.info("═" * 60)
        logger.info("PipelineStageWorker.run() — stage='%s'", self.stage_name)
        logger.info("═" * 60)
        logger.info("  Config project_root: %s",
                     self.config.project_root if self.config else "None")
        for k, v in self._kwargs.items():
            logger.info("  kwarg %s = %s", k, v)

        self._cancel = False
        self._cancel_event.clear()

        try:
            from gui.search_runner import AsyncBridge
            from gui.stage_runner import _STAGE_REGISTRY

            if self.stage_name not in _STAGE_REGISTRY:
                msg = f"Unknown stage: '{self.stage_name}'"
                logger.error(msg)
                self.error.emit(self.stage_name, msg)
                return

            stage_func, is_async = _STAGE_REGISTRY[self.stage_name]

            def progress_cb(
                payload: str | dict[str, object] | None,
            ) -> None:
                line = _progress_message(payload)
                logger.info("Stage '%s' progress: %s", self.stage_name, line)
                self.progress.emit(line)
                counts = _progress_count_values(payload)
                if counts is not None:
                    self.progress_counts.emit(*counts)

            if is_async:
                bridge = AsyncBridge()
                try:
                    coro = stage_func(
                        self.config,
                        cancel_event=self._cancel_event,
                        progress_callback=progress_cb,
                        **self._kwargs,
                    )
                    future = bridge.submit(coro)
                    result = future.result()
                finally:
                    bridge.shutdown()
            else:
                result = stage_func(
                    self.config,
                    cancel_event=self._cancel_event,
                    **self._kwargs,
                )

            if self._cancel:
                logger.info("Stage '%s' was cancelled", self.stage_name)
                self.cancelled.emit()
                return

            logger.info("PipelineStageWorker: stage '%s' completed, result type=%s",
                        self.stage_name, type(result).__name__)
            self.finished.emit(self.stage_name, result)

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("PipelineStageWorker ERROR (stage='%s'): %s\n%s",
                         self.stage_name, exc, tb)
            self.error.emit(self.stage_name, f"Stage '{self.stage_name}' failed: {exc}")
