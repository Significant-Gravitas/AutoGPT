If the goal oriented task pipeline fails, it means: 
- you somehow changed the way the system prompt is generated 
- or you broke autogpt.

To know which one, you can run the following command: 
```bash
pytest -s -k tests/integration/goal_oriented

If the test is successful, it will record new cassettes in VCR. Then you can just push these to your branch and the pipeline
will pass
