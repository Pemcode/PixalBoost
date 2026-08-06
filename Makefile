# Thin delegator to the canonical task runner (`uv run poe <task>`).
# Kept so the `make check` convention works on CI and RunPod, where GNU make
# exists. On Windows dev machines, call `uv run poe check` directly.
# See DECISIONS.md ADR-0002.

.PHONY: setup lint fmt typecheck test check

setup lint fmt typecheck test check:
	uv run poe $@
