# Pre-trained reaction models

Repo containing pre-trained reactions models wrapped with syntheseus.

| :exclamation:  Note: |
|----------------------|

This repository is largely superseded by the [syntheseus-retro-star-benchmark](https://github.com/AustinT/syntheseus-retro-star-benchmark) package.
It is recommended to use this instead.

## Development

### Installation

Main dependency is `syntheseus`, everything else depends on the specific reaction model.

### Formatting

Use pre-commit to enforce formatting, large file checks, etc.

If not already installed in your environment, run:

```bash
conda install pre-commit
```

To install the precommit hooks:

```bash
pre-commit install
```

Now a series of useful checks will be run before any commit.

### Testing

Run tests with:

```bash
python -m pytest tests/
```
