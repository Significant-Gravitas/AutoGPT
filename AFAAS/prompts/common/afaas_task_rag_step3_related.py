from __future__ import annotations

from AFAAS.interfaces.prompts.base_task_rag import BaseTaskRagStrategyConfiguration,  BaseTaskRagStrategy

class AfaasTaskRagStep3StrategyConfiguration(BaseTaskRagStrategyConfiguration) :
    task_context_length: int = 200

class AfaasTaskRagStep3Strategy(BaseTaskRagStrategy):
    STRATEGY_NAME = "afaas_task_rag_step3_history"
    default_configuration = AfaasTaskRagStep3StrategyConfiguration()
