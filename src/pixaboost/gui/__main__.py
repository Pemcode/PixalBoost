"""Entry point for ``python -m pixaboost.gui`` and ``pixaboost-gui``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtCore import QLibraryInfo, QLocale, QTimer, QTranslator
from PyQt6.QtWidgets import QApplication

from pixaboost.gui.main_window import MainWindow
from pixaboost.gui.remote_trial import RemoteTrialDefaults, project_git_sha
from pixaboost.gui.single_view_adapter import ExistingPodSingleViewRunner
from pixaboost.gui.theme import STYLESHEET


def install_french(app: QApplication) -> QTranslator | None:
    """Install Qt's French standard-dialog translations when available."""
    translator = QTranslator()
    translations = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if translator.load(QLocale(QLocale.Language.French), "qtbase", "_", translations):
        app.installTranslator(translator)
        return translator
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="create the real window and exit immediately (offscreen-compatible)",
    )
    parser.add_argument("--trial-image", type=Path, help="prefill the single-view image")
    parser.add_argument("--ssh-host", help="prefill the existing Pod SSH host")
    parser.add_argument("--ssh-user", help="prefill the existing Pod SSH user")
    parser.add_argument("--ssh-key", type=Path, help="prefill the SSH private-key path")
    parser.add_argument("--ssh-known-hosts", type=Path, help="prefill the strict known_hosts path")
    parser.add_argument("--pixal3d-sha", help="prefill the expected Pixal3D Git SHA")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    existing = QApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        raise RuntimeError("a non-GUI Qt application already exists")
    app = existing if isinstance(existing, QApplication) else QApplication(sys.argv[:1])
    app.setApplicationName("PixaBoost")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    translator = install_french(app)  # noqa: F841 - Qt does not retain it for us
    repo_root = Path(__file__).resolve().parents[3]
    defaults = RemoteTrialDefaults.from_environment().merged(
        image_path=args.trial_image,
        host=args.ssh_host,
        username=args.ssh_user,
        private_key_path=args.ssh_key,
        known_hosts_path=args.ssh_known_hosts,
        expected_pixal3d_sha=args.pixal3d_sha,
        project_git_sha=project_git_sha(repo_root),
    )
    window = MainWindow(
        repo_root=repo_root,
        remote_runner_factory=lambda: ExistingPodSingleViewRunner(repo_root),
        remote_defaults=defaults,
    )
    window.show()
    if args.smoke_test:
        QTimer.singleShot(50, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
