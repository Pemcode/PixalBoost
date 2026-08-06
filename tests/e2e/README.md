# tests/e2e

GPU- and network-bound tests. **Never run in CPU CI** and never part of `poe check`.

Every test here must carry `@pytest.mark.gpu` or `@pytest.mark.network`.
Run them explicitly, against a RunPod pod:

```bash
uv run pytest tests/e2e -m gpu
```
