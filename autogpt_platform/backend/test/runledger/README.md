# RunLedger PoC (backend blocks)

This PoC runs a deterministic RunLedger suite against the Google Docs formatting block.
It uses a mocked Google Docs service (no network, no secrets) and asserts the expected RGB output.

Run locally (from autogpt_platform/backend):

```
poetry run python -m runledger run test/runledger --mode live
```
