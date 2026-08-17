# scripts

Dev / bench tooling. The canonical build scripts (`build-clean-*`, `ab_env`,
`run-tests`, `install-hooks`, ...) stay one level up at the clean-stack root
because the docs reference them there; this directory holds the rest.

- `byron_stratum_probe.py` -- standalone stratum-protocol diagnostic probe.
- `vast/`  -- multi-card vast.ai benchmarking suite (see `vast/README.md`).
- `attic/` -- retired one-off build/probe scripts (see `attic/README.md`).
