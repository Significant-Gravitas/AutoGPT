# Challenge Definitions

This directory contains challenge data files used by the `direct_benchmark` harness.

Each challenge is a directory containing a `data.json` file that defines the task, ground truth, and evaluation criteria. See `CHALLENGE.md` for the data schema.

## Structure

```
challenges/
├── abilities/          # Basic agent capabilities (read/write files)
├── alignment/          # Safety and alignment tests
├── verticals/          # Domain-specific challenges (code, data, scrape, etc.)
└── library/            # Additional challenge library
```

## Running Challenges

```bash
# From the classic/ directory
poetry run direct-benchmark run --tests ReadFile
poetry run direct-benchmark run --strategies one_shot --models claude
poetry run direct-benchmark run --help
```
