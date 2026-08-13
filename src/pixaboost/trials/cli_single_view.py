"""Argparse wiring for the public single-view trial service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pixaboost.backends.cache import ArtifactCache
from pixaboost.backends.pixal3d import GenerationParams
from pixaboost.backends.ssh_pod import ExistingPodUseApproval, SshPodConfig, SshPodError
from pixaboost.observability import TelemetryEvent, encode_event
from pixaboost.trials.single_view import SingleViewTrialConfig, run_single_view_trial

EXIT_OK = 0
EXIT_APPROVAL_REQUIRED = 3
EXIT_REMOTE_FAILURE = 4
EXIT_INVALID_INPUT = 5


def configure_single_view_cli(parser: argparse.ArgumentParser) -> None:
    """Register ``reconstruct single-view`` on the root parser."""
    commands = parser.add_subparsers(dest="command")
    reconstruct = commands.add_parser("reconstruct", help="Run a reconstruction trial")
    modes = reconstruct.add_subparsers(dest="reconstruction_mode")
    single = modes.add_parser("single-view", help="Reconstruct one image into a cached GLB")
    single.add_argument("image", type=Path)
    single.add_argument("--backend", choices=("ssh-pod",), required=True)
    single.add_argument("--host", required=True)
    single.add_argument("--user", required=True)
    single.add_argument("--key", type=Path, required=True)
    single.add_argument("--known-hosts", type=Path, required=True)
    single.add_argument("--revision", required=True, help="Expected full Pixal3D Git SHA")
    single.add_argument("--project-sha", required=True, help="Current full PixaBoost Git SHA")
    single.add_argument("--cache", type=Path, default=Path("artifacts"))
    single.add_argument("--runs", type=Path, default=Path("runs"))
    single.add_argument("--port", type=int, default=22)
    single.add_argument("--remote-root", default="/workspace/pixaboost-jobs")
    single.add_argument("--seed", type=int, default=42)
    single.add_argument("--resolution", type=int, default=-1)
    single.add_argument("--fov", type=float, default=-1.0)
    single.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    single.add_argument(
        "--confirm-existing-pod-use",
        action="store_true",
        help="One-shot confirmation for a cache miss; never provisions or buys credits",
    )


def dispatch_single_view_cli(args: argparse.Namespace) -> int | None:
    """Execute the registered command, or return ``None`` for another command."""
    if args.command != "reconstruct" or args.reconstruction_mode != "single-view":
        return None
    try:
        ssh = SshPodConfig(
            host=args.host,
            username=args.user,
            private_key_path=args.key,
            known_hosts_path=args.known_hosts,
            expected_pixal3d_sha=args.revision,
            project_git_sha=args.project_sha,
            local_runs_root=args.runs,
            remote_root=args.remote_root,
            port=args.port,
        )
        config = SingleViewTrialConfig(
            image_path=args.image,
            params=GenerationParams(
                seed=args.seed,
                resolution=args.resolution,
                low_vram=args.low_vram,
                fov=args.fov,
            ),
            cache=ArtifactCache(args.cache),
            ssh=ssh,
            poses=({"view": "input", "transform": None},),
        )
        approval = ExistingPodUseApproval.grant(ssh) if args.confirm_existing_pod_use else None
        run_single_view_trial(config, approval=approval, event_sink=_print_event)
        return EXIT_OK
    except SshPodError as error:
        print(f"pixaboost: {error}", file=sys.stderr)
        return EXIT_APPROVAL_REQUIRED if error.code == "approval_required" else EXIT_REMOTE_FAILURE
    except (OSError, ValueError) as error:
        print(f"pixaboost: {error}", file=sys.stderr)
        return EXIT_INVALID_INPUT


def _print_event(event: TelemetryEvent) -> None:
    print(encode_event(event), flush=True)
