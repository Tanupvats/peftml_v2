# Contributing to peftml

Thanks for your interest in contributing! Here's how to get started.

## Development setup

```bash
git clone https://github.com/peftml/peftml.git
cd peftml
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=peftml --cov-report=term-missing
```

## Code style

We use **ruff** for linting and formatting:

```bash
ruff check peftml/ tests/
ruff format peftml/ tests/
```

Type checking with **mypy**:

```bash
mypy peftml/
```

## Adding a new compression method

1. Create a module in the appropriate subpackage (e.g. `peftml/quantization/my_method.py`).
2. Add a dataclass config to `peftml/core/config.py`.
3. Expose the new method in the subpackage `__init__.py` and the top-level `__init__.py`.
4. Add a convenience method to `ModelCompressor` in `peftml/pipelines/compressor.py`.
5. Write tests in `tests/test_<subpackage>.py`.
6. Update the README and CHANGELOG.

## Pull request checklist

- [ ] All existing tests pass (`pytest tests/ -v`).
- [ ] New code has tests with reasonable coverage.
- [ ] `ruff check` and `ruff format --check` pass.
- [ ] Docstrings on all public classes and functions.
- [ ] CHANGELOG.md updated.
