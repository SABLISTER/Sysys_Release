"""
Project manager for SystS — handles New/Open/Recent/Save project operations.

A project is a folder containing:
  - research_config.yaml   (pipeline config)
  - pipeline_config.json   (settings snapshot)
  - project_state.json     (window state + component state)
"""
from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QWidget

from core.models import HypothesisComponent
from core.domain_config import domain_config_snapshot
from core.io import atomic_write_json
from gui.settings_persistence import API_KEY_QSETTINGS_KEYS

logger = logging.getLogger(__name__)

_SETTINGS_ORG = "Sysys"
_SETTINGS_APP = "HypothesisConsensusAnalyzer"
_RECENT_KEY = "projects/recent"
_MAX_RECENT = 8


class ProjectManager:
    """Manages project folder lifecycle: create, open, save, recent list."""

    BACKUP_MANIFEST = "backup_manifest.json"
    BACKUP_FORMAT_VERSION = 1
    AUTO_BACKUP_FORMAT = "systs_incremental_backup"
    AUTO_BACKUP_FOLDER_NAME = "SystS Backups"

    @staticmethod
    def serialize_hypothesis_components(components: list) -> list[dict]:
        """Serialize Search-tab components into JSON-safe project state."""
        rows: list[dict] = []
        for component in components or []:
            rows.append({
                "id": str(getattr(component, "id", "")),
                "label": str(getattr(component, "label", "")),
                "description": str(getattr(component, "description", "")),
                "keywords": [
                    str(keyword)
                    for keyword in (getattr(component, "keywords", []) or [])
                    if str(keyword).strip()
                ],
                "weight": float(getattr(component, "weight", 1.0) or 0.0),
            })
        return rows

    @staticmethod
    def deserialize_hypothesis_components(rows: object) -> list[HypothesisComponent]:
        """Rebuild Search-tab components from saved project state."""
        components: list[HypothesisComponent] = []
        if not isinstance(rows, list):
            return components
        for row in rows:
            if not isinstance(row, dict):
                continue
            comp_id = str(row.get("id") or "").strip()
            label = str(row.get("label") or "").strip()
            if not comp_id or not label:
                continue
            keywords = [
                str(keyword).strip()
                for keyword in (row.get("keywords") or [])
                if str(keyword).strip()
            ]
            try:
                weight = float(row.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            components.append(HypothesisComponent(
                id=comp_id,
                label=label,
                description=str(row.get("description") or ""),
                keywords=keywords,
                weight=weight,
            ))
        return components

    # ── New project ───────────────────────────────────────────────────

    @staticmethod
    def new_project(parent_widget: Optional[QWidget] = None) -> Optional[Path]:
        """Prompt user to choose a folder; create project skeleton inside it.

        Returns the project Path on success, or None if cancelled/error.
        """
        folder = QFileDialog.getExistingDirectory(
            parent_widget,
            "Choose folder for new project",
            str(Path.home()),
        )
        if not folder:
            return None

        project_path = Path(folder)
        try:
            ProjectManager._create_skeleton(project_path)
            ProjectManager.add_recent_project(project_path)
            logger.info(f"New project created at {project_path}")
            return project_path
        except Exception as exc:
            logger.error(f"Failed to create project at {project_path}: {exc}")
            return None

    @staticmethod
    def _create_skeleton(project_path: Path) -> None:
        """Create required sub-directories and placeholder files."""
        subdirs = [
            "corpus", "claims", "reports", "fulltext", "exports",
            "data/raw", "data/corpus", "data/claims", "data/clusters",
            "output",
        ]
        for sd in subdirs:
            (project_path / sd).mkdir(parents=True, exist_ok=True)

        # Write a minimal research_config.yaml if absent
        cfg_yaml = project_path / "research_config.yaml"
        if not cfg_yaml.exists():
            cfg_yaml.write_text(
                "# SystS Research Configuration\n"
                "project_root: .\n"
                "llm_provider: ollama\n",
                encoding="utf-8",
            )

        # Write a minimal pipeline_config.json if absent
        cfg_json = project_path / "pipeline_config.json"
        if not cfg_json.exists():
            cfg_json.write_text(
                json.dumps({"project_root": str(project_path)}, indent=2),
                encoding="utf-8",
            )

    # ── Open project ─────────────────────────────────────────────────

    @staticmethod
    def open_project(parent_widget: Optional[QWidget] = None) -> Optional[Path]:
        """Prompt user to select an existing project folder.

        Returns the project Path on success, or None if cancelled/error.
        """
        folder = QFileDialog.getExistingDirectory(
            parent_widget,
            "Open SystS project folder",
            str(Path.home()),
        )
        if not folder:
            return None

        project_path = Path(folder)
        if not project_path.is_dir():
            logger.error(f"Selected path is not a directory: {project_path}")
            return None

        validation = ProjectManager.validate_project(project_path)
        if not validation["valid"]:
            logger.warning(
                "Selected folder does not look like a SystS project: %s",
                project_path,
            )

        ProjectManager.add_recent_project(project_path)
        logger.info(f"Opened project at {project_path}")
        return project_path

    @staticmethod
    def validate_project(project_path: Path) -> dict:
        """Return basic validity and artifact information for a project folder."""
        project_path = Path(project_path)
        markers = [
            "research_config.yaml",
            "research_config.yml",
            "research_config.json",
            "pipeline_config.json",
            "project_state.json",
        ]
        artifacts = {
            "research_config": any((project_path / name).exists() for name in markers[:3]),
            "pipeline_config": (project_path / "pipeline_config.json").exists(),
            "project_state": (project_path / "project_state.json").exists(),
            "data_dir": (project_path / "data").is_dir(),
            "output_dir": (project_path / "output").is_dir(),
            "corpus_json": (project_path / "data" / "corpus" / "corpus.json").exists(),
            "relevant_json": (project_path / "data" / "claims" / "relevant.json").exists(),
            "claims_json": (project_path / "data" / "claims" / "claims.json").exists(),
        }
        valid = project_path.is_dir() and (
            any((project_path / name).exists() for name in markers)
            or artifacts["data_dir"]
            or artifacts["output_dir"]
            or (project_path / "corpus").is_dir()
            or (project_path / "claims").is_dir()
        )
        warnings = []
        if project_path.is_dir() and not artifacts["research_config"]:
            warnings.append("No research_config file found.")
        if project_path.is_dir() and not artifacts["data_dir"] and not (project_path / "corpus").is_dir():
            warnings.append("No data or corpus directory found yet.")
        return {"valid": valid, "artifacts": artifacts, "warnings": warnings}

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _should_skip_backup_path(path: Path) -> bool:
        skip_names = {"__pycache__", ".DS_Store", ".git"}
        if any(part in skip_names for part in path.parts):
            return True
        return path.name.endswith((".tmp", ".partial", ".lock"))

    @staticmethod
    def _artifact_presence(project_path: Path) -> dict:
        return {
            "corpus": (project_path / "data" / "corpus" / "corpus.json").exists(),
            "relevant": (project_path / "data" / "claims" / "relevant.json").exists(),
            "claims": (project_path / "data" / "claims" / "claims.json").exists(),
            "snapshots": any((project_path / "output").glob("**/snapshots/*.json"))
            if (project_path / "output").exists() else False,
            "reports": any((project_path / "output").glob("**/*.md"))
            if (project_path / "output").exists() else False,
        }

    @staticmethod
    def _safe_backup_name(name: str) -> str:
        return "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in str(name or "").strip()
        ) or "SystS_Project"

    @staticmethod
    def default_auto_backup_destination() -> Path:
        """Return a good default folder for synced automatic backups."""
        home = Path.home()
        cloud_root = home / "Library" / "CloudStorage"
        if cloud_root.exists():
            for candidate in sorted(cloud_root.glob("GoogleDrive-*")):
                my_drive = candidate / "My Drive"
                if my_drive.is_dir():
                    return my_drive / ProjectManager.AUTO_BACKUP_FOLDER_NAME
                if candidate.is_dir():
                    return candidate / ProjectManager.AUTO_BACKUP_FOLDER_NAME
        for candidate in (home / "Google Drive", home / "My Drive"):
            if candidate.is_dir():
                return candidate / ProjectManager.AUTO_BACKUP_FOLDER_NAME
        return home / ProjectManager.AUTO_BACKUP_FOLDER_NAME

    @staticmethod
    def _auto_backup_destination(project_path: Path, destination_dir: Path) -> Path:
        return destination_dir / ProjectManager._safe_backup_name(project_path.name) / "latest"

    @staticmethod
    def _validate_backup_destination(project_path: Path, destination_dir: Path) -> None:
        if project_path.resolve() == destination_dir.resolve():
            raise ValueError("Backup destination cannot be the project folder itself.")
        if ProjectManager._is_relative_to(destination_dir, project_path):
            raise ValueError("Backup destination cannot be inside the project folder.")

    @staticmethod
    def _read_backup_manifest(path: Path) -> dict:
        manifest_path = path / ProjectManager.BACKUP_MANIFEST
        if not manifest_path.exists():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def resolve_backup_source(backup_path: Path) -> Path:
        """Resolve zip, latest/, or project auto-backup container into restorable content."""
        backup_path = Path(backup_path).expanduser().resolve()
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        if backup_path.is_file():
            if zipfile.is_zipfile(backup_path):
                return backup_path
            raise ValueError("Backup must be a .zip file or a backup folder.")
        if ProjectManager.validate_project(backup_path)["valid"]:
            return backup_path

        manifest = ProjectManager._read_backup_manifest(backup_path)
        if manifest.get("format") == ProjectManager.AUTO_BACKUP_FORMAT:
            return backup_path

        latest = backup_path / "latest"
        if latest.is_dir():
            latest_manifest = ProjectManager._read_backup_manifest(latest)
            if (
                latest_manifest.get("format") == ProjectManager.AUTO_BACKUP_FORMAT
                or ProjectManager.validate_project(latest)["valid"]
            ):
                return latest

        candidates: list[Path] = []
        for child in backup_path.iterdir():
            latest_child = child / "latest"
            if not latest_child.is_dir():
                continue
            latest_manifest = ProjectManager._read_backup_manifest(latest_child)
            if latest_manifest.get("format") == ProjectManager.AUTO_BACKUP_FORMAT:
                candidates.append(latest_child)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                "Backup folder contains multiple project backups; choose one project's latest folder."
            )
        raise ValueError("Backup does not contain a recognizable SystS project.")

    @staticmethod
    def backup_metadata(backup_path: Path) -> dict:
        """Return backup manifest metadata when available."""
        source = ProjectManager.resolve_backup_source(backup_path)
        if source.is_dir():
            return ProjectManager._read_backup_manifest(source)
        if zipfile.is_zipfile(source):
            try:
                with zipfile.ZipFile(source) as zf:
                    with zf.open(ProjectManager.BACKUP_MANIFEST) as fh:
                        data = json.loads(fh.read().decode("utf-8"))
            except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError):
                return {}
            return data if isinstance(data, dict) else {}
        return {}

    @staticmethod
    def _should_include_auto_backup_file(
        rel_path: Path,
        *,
        include_fulltext: bool,
        include_images: bool,
        max_file_bytes: int,
        file_size: int,
    ) -> bool:
        if file_size > max_file_bytes:
            return False

        skip_parts = {"__pycache__", ".git"}
        if any(part in skip_parts for part in rel_path.parts):
            return False
        if rel_path.name == ".DS_Store" or rel_path.name.endswith((".tmp", ".partial", ".lock")):
            return False

        suffix = rel_path.suffix.lower()
        if suffix in {".pdf", ".zip", ".sqlite", ".db"} and not include_fulltext:
            return False

        parts = set(rel_path.parts)
        heavy_fulltext_parts = {"source_archive", "paperqa", "fulltext_corpus"}
        if parts & heavy_fulltext_parts and not include_fulltext:
            return False

        critical_names = {
            "research_config.yaml", "research_config.yml", "research_config.json",
            "pipeline_config.json", "project_state.json",
        }
        if rel_path.name in critical_names:
            return True

        text_exts = {".json", ".yaml", ".yml", ".md", ".csv", ".txt", ".bib", ".ris", ".xml"}
        image_exts = {".png", ".svg", ".jpg", ".jpeg"}
        fulltext_exts = {".pdf", ".html", ".htm"}
        allowed_exts = set(text_exts)
        if include_images:
            allowed_exts |= image_exts
        if include_fulltext:
            allowed_exts |= fulltext_exts

        if suffix not in allowed_exts:
            return False

        top_level = rel_path.parts[0] if rel_path.parts else ""
        if top_level in {"data", "output", "corpus", "claims", "reports", "exports", "fulltext"}:
            return True
        return False

    @staticmethod
    def collect_auto_backup_files(
        project_path: Path,
        *,
        include_fulltext: bool = False,
        include_images: bool = True,
        max_file_mb: int = 100,
        destination_dir: Path | None = None,
    ) -> list[Path]:
        """Collect lightweight project files suitable for frequent auto-backup."""
        project_path = Path(project_path).expanduser().resolve()
        max_file_bytes = max(1, int(max_file_mb)) * 1024 * 1024
        files: list[Path] = []
        destination_dir = Path(destination_dir).expanduser().resolve() if destination_dir else None
        for path in project_path.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if destination_dir and ProjectManager._is_relative_to(destination_dir, project_path):
                if ProjectManager._is_relative_to(resolved, destination_dir):
                    continue
            try:
                rel_path = path.relative_to(project_path)
                file_size = path.stat().st_size
            except OSError:
                continue
            if ProjectManager._should_include_auto_backup_file(
                rel_path,
                include_fulltext=include_fulltext,
                include_images=include_images,
                max_file_bytes=max_file_bytes,
                file_size=file_size,
            ):
                files.append(path)
        return files

    @staticmethod
    def _file_needs_copy(src: Path, dest: Path) -> bool:
        if not dest.exists():
            return True
        try:
            src_stat = src.stat()
            dest_stat = dest.stat()
        except OSError:
            return True
        if src_stat.st_size != dest_stat.st_size:
            return True
        return src_stat.st_mtime_ns > dest_stat.st_mtime_ns

    @staticmethod
    def sync_project_backup(
        project_path: Path,
        destination_dir: Path,
        *,
        include_fulltext: bool = False,
        include_images: bool = True,
        max_file_mb: int = 100,
    ) -> dict:
        """Incrementally mirror changed project files into a Drive-friendly backup folder."""
        project_path = Path(project_path).expanduser().resolve()
        destination_dir = Path(destination_dir).expanduser().resolve()
        if not project_path.is_dir():
            raise FileNotFoundError(f"Project folder not found: {project_path}")
        ProjectManager._validate_backup_destination(project_path, destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        backup_root = ProjectManager._auto_backup_destination(project_path, destination_dir)
        backup_root.mkdir(parents=True, exist_ok=True)

        files = ProjectManager.collect_auto_backup_files(
            project_path,
            include_fulltext=include_fulltext,
            include_images=include_images,
            max_file_mb=max_file_mb,
            destination_dir=destination_dir,
        )

        copied = 0
        skipped = 0
        total_bytes = 0
        rel_paths: set[str] = set()
        for src in files:
            rel = src.relative_to(project_path)
            rel_name = rel.as_posix()
            rel_paths.add(rel_name)
            total_bytes += src.stat().st_size
            dest = backup_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if ProjectManager._file_needs_copy(src, dest):
                temp_dest = dest.with_name(dest.name + ".tmp")
                shutil.copy2(src, temp_dest)
                temp_dest.replace(dest)
                copied += 1
            else:
                skipped += 1

        stale_removed = 0
        for existing in backup_root.rglob("*"):
            if not existing.is_file() or existing.name == ProjectManager.BACKUP_MANIFEST:
                continue
            try:
                rel = existing.relative_to(backup_root).as_posix()
            except ValueError:
                continue
            if rel not in rel_paths:
                existing.unlink(missing_ok=True)
                stale_removed += 1

        manifest = {
            "format": ProjectManager.AUTO_BACKUP_FORMAT,
            "format_version": ProjectManager.BACKUP_FORMAT_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_name": project_path.name,
            "source_project_path": str(project_path),
            "backup_root": str(backup_root),
            "file_count": len(files),
            "copied": copied,
            "unchanged": skipped,
            "stale_removed": stale_removed,
            "total_bytes": total_bytes,
            "options": {
                "include_fulltext": include_fulltext,
                "include_images": include_images,
                "max_file_mb": max_file_mb,
            },
            "artifacts": ProjectManager._artifact_presence(project_path),
        }
        manifest_path = backup_root / ProjectManager.BACKUP_MANIFEST
        temp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temp_manifest.replace(manifest_path)

        logger.info(
            "Project auto-backup synced: %s (%d copied, %d unchanged)",
            backup_root,
            copied,
            skipped,
        )
        return manifest

    @staticmethod
    def backup_project(project_path: Path, destination_dir: Path) -> Path:
        """Create a timestamped zip backup of a project folder."""
        project_path = Path(project_path).expanduser().resolve()
        destination_dir = Path(destination_dir).expanduser().resolve()
        if not project_path.is_dir():
            raise FileNotFoundError(f"Project folder not found: {project_path}")
        ProjectManager._validate_backup_destination(project_path, destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = ProjectManager._safe_backup_name(project_path.name)
        final_path = destination_dir / f"SystS_Backup_{safe_name}_{timestamp}.zip"
        counter = 2
        while final_path.exists():
            final_path = destination_dir / f"SystS_Backup_{safe_name}_{timestamp}_{counter}.zip"
            counter += 1
        temp_path = final_path.with_suffix(final_path.suffix + ".tmp")

        files: list[Path] = []
        for path in project_path.rglob("*"):
            if not path.is_file() or ProjectManager._should_skip_backup_path(path):
                continue
            resolved = path.resolve()
            if resolved in {temp_path, final_path}:
                continue
            if (
                ProjectManager._is_relative_to(destination_dir, project_path)
                and ProjectManager._is_relative_to(resolved, destination_dir)
            ):
                continue
            files.append(path)

        total_bytes = sum(path.stat().st_size for path in files)
        manifest = {
            "format": "systs_project_backup",
            "format_version": ProjectManager.BACKUP_FORMAT_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_name": project_path.name,
            "source_project_path": str(project_path),
            "file_count": len(files),
            "total_bytes": total_bytes,
            "artifacts": ProjectManager._artifact_presence(project_path),
        }

        try:
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(ProjectManager.BACKUP_MANIFEST, json.dumps(manifest, indent=2))
                for path in files:
                    zf.write(path, path.relative_to(project_path).as_posix())
            temp_path.replace(final_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        logger.info("Project backup created: %s", final_path)
        return final_path

    @staticmethod
    def _safe_extract_zip(zip_path: Path, destination_dir: Path) -> None:
        destination_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe backup member path: {member.filename}")
                zf.extract(member, destination_dir)

    @staticmethod
    def restore_project(backup_path: Path, destination_path: Path, overwrite: bool = False) -> Path:
        """Restore a project backup zip or folder into destination_path."""
        backup_path = ProjectManager.resolve_backup_source(backup_path)
        destination_path = Path(destination_path).expanduser().resolve()

        if destination_path.exists() and any(destination_path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Destination is not empty: {destination_path}")
            shutil.rmtree(destination_path)

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination_path.parent / (
            f".{destination_path.name}.restore_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if temp_path.exists():
            shutil.rmtree(temp_path)

        try:
            if backup_path.is_dir():
                shutil.copytree(backup_path, temp_path)
            elif zipfile.is_zipfile(backup_path):
                ProjectManager._safe_extract_zip(backup_path, temp_path)
            else:
                raise ValueError("Backup must be a .zip file or a backup folder.")

            validation = ProjectManager.validate_project(temp_path)
            if not validation["valid"]:
                raise ValueError("Backup does not contain a recognizable SystS project.")

            if destination_path.exists():
                shutil.rmtree(destination_path)
            temp_path.rename(destination_path)
        except Exception:
            if temp_path.exists():
                shutil.rmtree(temp_path)
            raise

        ProjectManager.add_recent_project(destination_path)
        logger.info("Project restored to: %s", destination_path)
        return destination_path

    # ── Save / Load project state ─────────────────────────────────────

    @staticmethod
    def save_project(
        project_path: Path,
        config: object,
        window_state: dict,
    ) -> None:
        """Write project_state.json to the project folder.

        Parameters
        ----------
        project_path:
            Root folder of the project.
        config:
            The active ``Config`` object (attributes serialised to dict).
        window_state:
            Arbitrary dict of window/component state to persist.
        """
        state_path = project_path / "project_state.json"
        # Serialize config fields that are JSON-safe
        cfg_dict: dict = {}
        if config is not None:
            for attr in dir(config):
                if (
                    attr.startswith("_")
                    or attr.endswith("_api_key")
                    or (attr in API_KEY_QSETTINGS_KEYS and attr != "pubmed_email")
                ):
                    continue
                try:
                    val = getattr(config, attr)
                    if callable(val):
                        continue
                    # Keep only primitives / paths
                    if isinstance(val, (str, int, float, bool, type(None))):
                        cfg_dict[attr] = val
                    elif isinstance(val, Path):
                        cfg_dict[attr] = str(val)
                except Exception:
                    pass

        payload = {
            "config": cfg_dict,
            "window_state": window_state,
        }
        if config is not None:
            payload["domain_config"] = domain_config_snapshot(config)
        try:
            atomic_write_json(state_path, payload)
            logger.info(f"Project saved to {state_path}")
        except Exception as exc:
            logger.error(f"Failed to save project: {exc}")

    @staticmethod
    def load_project(project_path: Path) -> dict:
        """Read project_state.json and return its contents as a dict.

        Returns an empty dict if the file doesn't exist or is malformed.
        """
        state_path = project_path / "project_state.json"
        if not state_path.exists():
            return {}
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Could not load project state: {exc}")
            return {}

    # ── Recent projects list ──────────────────────────────────────────

    @staticmethod
    def get_recent_projects() -> list[Path]:
        """Return up to _MAX_RECENT recently used project paths (as Path objects)."""
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        raw = cast(list, s.value(_RECENT_KEY, [], type=list))
        paths: list[Path] = []
        for item in raw:
            p = Path(str(item))
            if p.is_dir():
                paths.append(p)
        return paths[:_MAX_RECENT]

    @staticmethod
    def add_recent_project(path: Path) -> None:
        """Prepend *path* to the recent list (max _MAX_RECENT entries, no duplicates)."""
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        raw = cast(list, s.value(_RECENT_KEY, [], type=list))
        str_path = str(path.resolve())
        # Remove duplicates
        raw = [r for r in raw if str(r) != str_path]
        raw.insert(0, str_path)
        raw = raw[:_MAX_RECENT]
        s.setValue(_RECENT_KEY, raw)
