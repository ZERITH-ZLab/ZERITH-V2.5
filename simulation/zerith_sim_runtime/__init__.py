"""Dependency-light protocols shared with external simulator runtimes.

Modules in this package must remain importable without the ZERITH 2.5 training
stack. In particular, the sealed LIBERO simulator environment intentionally
does not carry PyTorch 2.9, mmengine, or Megatron-Core.
"""
