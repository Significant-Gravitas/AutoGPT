### Background

<!-- Provide a brief overview of why this change is being made. Include any relevant context, prior discussions, or links to relevant issues. -->
Since each deployment is created according to one specific model in Azure OpanAI Service, the deployment id should be specific by model.

### Changes

<!-- Describe the changes made in this pull request. Be specific and detailed. -->
In `.env.template`, modify `OPENAI_DEPLOYMENT_ID` to `FAST_LLM_MODEL_DEPLOYMENT_ID` and `SMART_LLM_MODEL_DEPLOYMENT_ID`.
Add a function `get_openai_deployment_id` in `llm_utils.py` to specific deployment id by model (fast/smart).
Also updated README.

### Test Plan

<!-- Explain how you tested this functionality. Include the steps to reproduce and any relevant test cases. -->

### Change Safety

- [ ] I have added tests to cover my changes
- [x] I have considered potential risks and mitigations for my changes

<!-- If you haven't added tests, please explain why. If you have, check the appropriate box. -->
It more like a modification to fit API requirement, does not affect core functionality.
