from __future__ import annotations

from AFAAS.interfaces.prompts.base_task_rag import (
    BaseTaskRagStrategy,
    BaseTaskRagStrategyConfiguration,
)


class AfaasTaskRagStep2StrategyConfiguration(BaseTaskRagStrategyConfiguration):
    task_context_length: int = 300


class AfaasTaskRagStep2Strategy(BaseTaskRagStrategy):
    STRATEGY_NAME = "afaas_task_rag_step2_history"
    default_configuration = AfaasTaskRagStep2StrategyConfiguration()
