# Contributing

Thank you for helping improve TennisVAR. Keep contributions focused on the public method implementation, documentation, tests, and reproducibility tooling.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,train]'
ruff check .
pytest
```

## Pull requests

- Explain the behavioral change and its relationship to the paper method.
- Add or update tests for model, schema, or configuration changes.
- Keep public defaults aligned with `configs/tennisvar.yaml`.
- Do not commit videos, frames, annotations, weights, caches, generated results, credentials, or absolute local paths.
- Do not add third-party code unless its license and attribution are documented and compatible with this repository.

For substantial API or architecture changes, open an issue before implementation.
