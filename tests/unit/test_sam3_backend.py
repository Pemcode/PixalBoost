"""Contract for the SAM 3 adapter (F14).

No torch, no download, no GPU: the model is substituted, because what is being
tested is the translation layer, not Meta's weights. Real-weight behaviour
belongs in tests/e2e.

The credential tests exist for one reason: `facebook/sam3` is gated, so a token
now flows through this code, and a token that leaks into an exception message
ends up in `runs/<id>/logs.jsonl`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from pixaboost.backends.sam3 import (
    DEFAULT_CHECKPOINT,
    TOKEN_ENV_VARS,
    PointPrompt,
    Sam3Error,
    Sam3TrackerRunner,
    SegmentationResult,
    _select_best,
    load_hf_token,
)

FAKE_TOKEN = "hf_notarealtokenjustforthetest"


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------


def test_the_environment_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_file = tmp_path / "huggingface.env"
    token_file.write_text("hf_fromfile", encoding="utf-8")
    monkeypatch.setenv(TOKEN_ENV_VARS[0], FAKE_TOKEN)
    assert load_hf_token(token_file) == FAKE_TOKEN


@pytest.mark.parametrize("variable", TOKEN_ENV_VARS)
def test_every_documented_hub_variable_is_honoured(
    variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, FAKE_TOKEN)
    assert load_hf_token(Path("does-not-exist")) == FAKE_TOKEN


@pytest.mark.parametrize(
    "content",
    [
        FAKE_TOKEN,
        f"{FAKE_TOKEN}\n",
        f"HF_TOKEN={FAKE_TOKEN}\n",
        f"# a comment\n\nHUGGINGFACE_TOKEN='{FAKE_TOKEN}'\n",
    ],
)
def test_a_bare_token_or_a_name_value_pair_both_work(
    content: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real huggingface.env holds a bare hf_ token with no trailing newline."""
    for name in TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    token_file = tmp_path / "huggingface.env"
    token_file.write_text(content, encoding="utf-8")
    assert load_hf_token(token_file) == FAKE_TOKEN


def test_a_missing_file_explains_that_the_repository_is_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(Sam3Error) as error:
        load_hf_token(tmp_path / "absent.env")
    assert "gated" in str(error.value)
    assert DEFAULT_CHECKPOINT in str(error.value)


def test_an_empty_token_file_is_an_error_not_an_empty_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    token_file = tmp_path / "huggingface.env"
    token_file.write_text("# only a comment\n\n", encoding="utf-8")
    with pytest.raises(Sam3Error, match="no usable token"):
        load_hf_token(token_file)


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch, *, from_pretrained: object
) -> None:
    """Stand in for torch and transformers so the loader runs without either.

    Substituting both is what keeps this test inside `poe check` instead of
    skipping on a machine with no torch -- and a secret-redaction test that
    skips is a secret-redaction test that never ran.
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"  # type: ignore[attr-defined]
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Sam3TrackerModel = from_pretrained  # type: ignore[attr-defined]
    fake_transformers.Sam3TrackerProcessor = from_pretrained  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_no_error_message_ever_contains_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leaked token would be persisted verbatim into runs/<id>/logs.jsonl.

    The fake failure echoes the token in an Authorization header, which is
    exactly what a real 401 from the Hub does.
    """
    for name in TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    token_file = tmp_path / "huggingface.env"
    token_file.write_text(FAKE_TOKEN, encoding="utf-8")

    class Exploding:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> None:
            raise RuntimeError(f"401 Unauthorized: Bearer {FAKE_TOKEN}")

    _install_fake_runtime(monkeypatch, from_pretrained=Exploding)

    with pytest.raises(Sam3Error) as error:
        Sam3TrackerRunner(token_file=token_file, device="cpu").load()

    assert FAKE_TOKEN not in str(error.value)
    assert FAKE_TOKEN not in repr(error.value)
    assert "accept the SAM License" in str(error.value), "the fix must be actionable"


def test_a_missing_token_is_refused_before_any_download_is_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    attempts: list[object] = []

    class Recording:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> None:
            attempts.append(args)

    _install_fake_runtime(monkeypatch, from_pretrained=Recording)

    with pytest.raises(Sam3Error, match="gated"):
        Sam3TrackerRunner(token_file=tmp_path / "absent.env", device="cpu").load()
    assert attempts == [], "credentials must be resolved before touching the Hub"


# --------------------------------------------------------------------------
# mask selection
# --------------------------------------------------------------------------


def test_the_highest_scoring_candidate_wins() -> None:
    masks = np.zeros((3, 4, 4), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 1:3, 1:3] = True
    masks[2, 3, 3] = True

    result = _select_best(masks, [0.10, 0.97, 0.55])

    assert result.iou_score == pytest.approx(0.97)
    assert np.array_equal(result.mask, masks[1])


def test_the_runner_up_scores_are_reported_best_first() -> None:
    masks = np.ones((3, 2, 2), dtype=bool)
    result = _select_best(masks, [0.2, 0.9, 0.5])
    assert result.candidate_scores == (0.9, 0.5, 0.2)


def test_a_near_tie_is_flagged_as_ambiguous() -> None:
    """A click on a wheel rim genuinely splits between 'the rim' and 'the wheel'."""
    assert SegmentationResult(np.ones((2, 2), bool), 0.90, (0.90, 0.88)).is_ambiguous
    assert not SegmentationResult(np.ones((2, 2), bool), 0.90, (0.90, 0.40)).is_ambiguous
    assert not SegmentationResult(np.ones((2, 2), bool), 0.90, (0.90,)).is_ambiguous


def test_an_all_empty_mask_is_an_error_not_a_silent_transparent_image() -> None:
    """An empty mask would compose to a fully transparent PNG and waste a GPU run."""
    with pytest.raises(Sam3Error, match="empty mask"):
        _select_best(np.zeros((2, 4, 4), dtype=bool), [0.9, 0.1])


def test_a_score_and_mask_count_mismatch_is_refused() -> None:
    with pytest.raises(Sam3Error, match="scores for"):
        _select_best(np.ones((3, 4, 4), dtype=bool), [0.9, 0.1])


def test_no_masks_at_all_is_refused() -> None:
    with pytest.raises(Sam3Error, match="no usable mask"):
        _select_best(np.zeros((0, 4, 4), dtype=bool), [])


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


def test_segmenting_without_a_prompt_is_refused_before_the_model_loads() -> None:
    """No prompt means no click; loading 3.4 GB to discover that is wasteful."""
    runner = Sam3TrackerRunner(token_file=Path("does-not-exist"))
    with pytest.raises(Sam3Error, match="at least one point prompt"):
        runner.segment(np.zeros((4, 4, 3), dtype=np.uint8), ())


def test_a_negative_prompt_is_expressible() -> None:
    """Excluding the clamp is a negative click, which is the whole point of SAM here."""
    assert PointPrompt(10, 20, positive=False).positive is False
    assert PointPrompt(10, 20).positive is True
