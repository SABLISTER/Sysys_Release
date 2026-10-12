# Verification requirements

Resolve every encountered test failure, collection/import error, and startup crash
before reporting the affected work complete. An unchanged failure count is useful
for diagnosis, but it is not a passing result. A failure's age or relationship to
the current edit does not excuse leaving it unresolved.

- Investigate the cause and fix the application, environment, or test fixture as
  appropriate. Keep the behavioral assertions when updating an outdated fixture.
- Do not weaken assertions, delete coverage, mark failures `xfail`, or skip tests
  to obtain a passing summary. Required dependencies must be installed and usable;
  missing dependencies are environment failures, not successful validation.
- Run the full repository test suite after the focused checks pass:

  ```sh
  venv/bin/python -m pytest tests/ -q
  ```

- GUI changes must also pass real MainWindow construction, event processing, and
  shutdown. This is included in `tests/test_main_window_startup.py` and can be run
  separately with:

  ```sh
  venv/bin/python -m pytest tests/test_main_window_startup.py -q
  ```

- Run `git diff --check` and appropriate syntax checks. Syntax checks alone do not
  validate runtime behavior.
- If an external blocker prevents a required check, identify it and report the
  work as incomplete. Do not relabel an unrun or failed check as passing.

## Headless GUI checks

The startup test runs in a subprocess with a clean temporary project, isolated
QSettings, writable cache directories, and `QT_QPA_PLATFORM=offscreen`. It builds
the real menus and tabs and exercises actual stage controls and corpus filtering.
Only the first-run welcome and automatic-backup prompts are suppressed.

Use ordinary no-op functions when replacing methods on `MainWindow` for such a
check. Bare `patch.object(MainWindow, "method")` creates a `MagicMock`; in the
installed PySide6 6.11 environment, those replacements cause Qt's dynamic signal
registration to segfault. For example:

```python
with patch.object(MainWindow, "_show_welcome_if_first_run", lambda self: None), \
     patch.object(MainWindow, "_maybe_prompt_auto_backup", lambda self, *args: None):
    window = MainWindow()
```

Do not stub menu/tab builders to make a startup check pass. Diagnose any crash,
including a crash caused by the test harness, and rerun real construction.

## Historical records and expected errors

Historical reports and archived plans may contain failed runs and
intermediate RED tests. They preserve diagnostic history; they do not authorize
accepting persistent failures. RED tests must reach GREEN before completion.
Tests that deliberately supply invalid input or simulate a failed service should
pass by asserting the expected error/fallback behavior.

Offline tests do not prove live API access, retrieval success, or scientific
validity. State those boundaries accurately without using them to excuse failures
in the checks that were actually run.
