"""Contract for the BiRefNet saliency adapter (F14).

The model is substituted: `poe check` downloads nothing. What is tested is the
thresholding decision and the preprocessing arithmetic, both of which are pure
numpy and both of which fail silently if wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixaboost.backends.birefnet import (
    DEFAULT_CHECKPOINT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    SaliencyError,
    _preprocess,
    _resize_nearest,
    threshold_saliency,
)


def test_the_checkpoint_is_the_mit_one_not_the_gated_rmbg() -> None:
    """ADR-0010: RMBG-2.0 is gated and `license: other`; this one is MIT."""
    assert DEFAULT_CHECKPOINT == "ZhengPeng7/BiRefNet"
    assert "rmbg" not in DEFAULT_CHECKPOINT.lower()


# --------------------------------------------------------------------------
# thresholding
# --------------------------------------------------------------------------


def test_scores_above_the_threshold_become_foreground() -> None:
    scores = np.array([[0.1, 0.9], [0.49, 0.51]])
    assert np.array_equal(threshold_saliency(scores, 0.5), [[False, True], [False, True]])


def test_an_all_background_map_is_an_explicit_error_not_an_empty_mask() -> None:
    """An empty mask would surface downstream as a confusing geometry error."""
    with pytest.raises(SaliencyError, match="no foreground"):
        threshold_saliency(np.full((8, 8), 0.02), 0.5)


def test_the_error_reports_the_best_score_so_the_threshold_can_be_judged() -> None:
    with pytest.raises(SaliencyError, match=r"0\.310"):
        threshold_saliency(np.full((4, 4), 0.31), 0.5)


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.2, 1.5])
def test_a_degenerate_threshold_is_refused(threshold: float) -> None:
    with pytest.raises(SaliencyError, match="strictly between"):
        threshold_saliency(np.full((4, 4), 0.6), threshold)


def test_a_non_two_dimensional_map_is_refused() -> None:
    with pytest.raises(SaliencyError, match="2-D"):
        threshold_saliency(np.zeros((4, 4, 3)))


# --------------------------------------------------------------------------
# preprocessing arithmetic
# --------------------------------------------------------------------------


def test_preprocessing_produces_channel_first_1024_square_input() -> None:
    image = np.full((300, 500, 3), 128, dtype=np.uint8)
    tensor = _preprocess(image)
    assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == np.float32


def test_normalisation_matches_the_imagenet_statistics_the_model_expects() -> None:
    """A wrong mean/std does not crash; it quietly degrades every mask."""
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    tensor = _preprocess(image)
    for channel in range(3):
        expected = (0.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert tensor[channel].mean() == pytest.approx(expected, abs=1e-4)


def test_a_white_image_normalises_to_the_expected_positive_value() -> None:
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    tensor = _preprocess(image)
    for channel in range(3):
        expected = (1.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert tensor[channel].mean() == pytest.approx(expected, abs=1e-4)


# --------------------------------------------------------------------------
# putting the score map back on the original grid
# --------------------------------------------------------------------------


def test_a_score_map_is_resized_back_to_the_photo_shape() -> None:
    scores = np.zeros((INPUT_SIZE, INPUT_SIZE))
    scores[: INPUT_SIZE // 2] = 1.0
    resized = _resize_nearest(scores, 200, 400)
    assert resized.shape == (200, 400)
    assert resized[:100].all() and not resized[100:].any()


def test_a_map_already_the_right_shape_is_returned_unchanged() -> None:
    scores = np.random.default_rng(0).random((30, 40))
    assert _resize_nearest(scores, 30, 40) is scores


def test_the_mask_a_caller_receives_has_the_photo_shape_not_the_model_shape() -> None:
    """The prompt is in photo pixels; a 1024-square mask would offset every click."""
    scores = np.zeros((INPUT_SIZE, INPUT_SIZE))
    scores[400:600, 400:600] = 0.99
    mask = threshold_saliency(_resize_nearest(scores, 120, 200))
    assert mask.shape == (120, 200)
    assert mask.any()
