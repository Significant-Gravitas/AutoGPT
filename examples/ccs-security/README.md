# CCS Security Integration for AutoGPT

[CCS](https://github.com/Correctover/ccs-verifier) provides sub-millisecond runtime verification for AutoGPT's command execution pipeline.

## Integration

```python
from examples.ccs_security.ccs_guard import CCSAutoGPTGuard

# Create guard and patch executor
guard = CCSAutoGPTGuard()
guard.patch_executor(code_executor_component)

# Now all shell/python execution is CCS-verified
# Dangerous commands are blocked before execution
```

## How It Works

CCS complements AutoGPT's existing allowlist/denylist with semantic analysis:
- Detects intent, not just patterns (e.g., `exec()` in dict key vs actual code exec)
- Blocks SSRF to cloud metadata, internal services
- Prevents credential/secret exfiltration
- Sub-millisecond verification (~7.5μs P50 in-process)

## Install

```bash
pip install ccs-verifier
```

## Reference

- [CCS IETF Draft](https://datatracker.ietf.org/doc/draft-correctover-ccs/)
- [CCS Zenodo DOI](https://doi.org/10.5281/zenodo.21783723)
