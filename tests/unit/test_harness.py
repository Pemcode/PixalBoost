"""Executable invariants for the repository's feature-list harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_LIST = REPO_ROOT / "feature_list.json"
ALLOWED_STATES = {"not_started", "active", "blocked", "passing"}


def load_features() -> list[dict[str, Any]]:
    try:
        payload = json.loads(FEATURE_LIST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        pytest.fail(
            "WHAT: feature_list.json is not valid JSON.\n"
            "WHY: the scheduler cannot trust task state when its source of truth is unreadable.\n"
            f"FIX: repair the JSON near line {error.lineno}, column {error.colno}: {error.msg}"
        )
    features = payload.get("features")
    assert isinstance(features, list), (
        "WHAT: feature_list.json has no features list.\n"
        "FIX: keep machine-readable feature entries under the top-level 'features' key."
    )
    return features


def test_feature_list_is_machine_readable_and_has_one_active_feature_at_most() -> None:
    features = load_features()

    ids = [feature.get("id") for feature in features]
    assert len(ids) == len(set(ids)), (
        "WHAT: feature ids are duplicated. FIX: use one unique id per feature."
    )

    active = []
    for feature in features:
        feature_id = feature.get("id", "<missing>")
        assert feature.get("state") in ALLOWED_STATES, (
            f"WHAT: {feature_id} has an invalid state. "
            f"FIX: choose one of {sorted(ALLOWED_STATES)}."
        )
        assert isinstance(feature.get("behavior"), str) and feature["behavior"].strip(), (
            f"WHAT: {feature_id} has no behavior contract. FIX: describe user-visible behavior."
        )
        assert isinstance(feature.get("verification"), str) and feature["verification"].strip(), (
            f"WHAT: {feature_id} has no verification command. FIX: add an executable command."
        )
        if feature["state"] == "active":
            active.append(feature_id)
        if feature["state"] == "passing":
            assert feature.get("evidence"), (
                f"WHAT: {feature_id} is passing without evidence. "
                "FIX: record the successful verification output."
            )
        if feature["state"] == "blocked":
            assert feature.get("blocked_on"), (
                f"WHAT: {feature_id} is blocked without a cause. "
                "FIX: record the blocking condition."
            )

    assert len(active) <= 1, (
        f"WHAT: multiple features are active: {active}. "
        "WHY: the repository enforces WIP=1. FIX: leave at most one feature active."
    )
