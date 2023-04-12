For now there aren't many tests.
So just run:

```
python tests/unit/json_tests.py
python tests/integration/memory_tests.py
```

paid test:
```
python tests/integration/test_commands.py 
```
This test costs 0.004$ per run with GPT-3.5. We will setup a pipeline in github action to allow people to run these tests for free.

The pipeline will be be triggered for now.

TODO: when we setup pytest, replace lines above by: pytest commands
