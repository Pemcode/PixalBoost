"""Pure CPU layer: geometry, metrics, rendering, registration.

Hard constraint: this package must never import torch, touch a GPU, or hit the
network. Enforced by tests/unit/test_architecture.py. See ARCHITECTURE.md.
"""
