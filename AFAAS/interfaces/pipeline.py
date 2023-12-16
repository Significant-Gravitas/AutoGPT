from __future__ import annotations
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine
from AFAAS.core.agents.routing.strategies import AutoCorrectionStrategy
from AFAAS.interfaces.task import AbstractTask
from AFAAS.interfaces.adapters import ChatModelResponse
from AFAAS.interfaces.job import JobInterface
from AFAAS.interfaces.agent import BaseLoop, BaseAgent


import copy
from typing import Any, Callable, Optional

class BasePipeline(BaseLoop):
    class LoophooksDict(BaseLoop.LoophooksDict):
        pass

    async def run(self, 
            agent: BaseAgent, 
            hooks: LoophooksDict, 
            user_input_handler: Callable[[str], Awaitable[str]] | None = None, 
            user_message_handler: Callable[[str], Awaitable[str]] | None = None) -> Coroutine[Any, Any, dict | None]:
        return super().run(agent, hooks, user_input_handler, user_message_handler)
    

class Pipeline(BasePipeline):
    def __init__(self, task :AbstractTask, agent : BaseAgent, autocorrection_strategy : AutoCorrectionStrategy = None , autocorrection_post_processing : Callable = None):

        super().__init__()
        self.jobs : list[JobInterface]= []
        #self._loop = agent._loop 
        self._agent : BaseAgent = agent
        self._task :AbstractTask = task
        self.autocorrection_strategy = autocorrection_strategy or AutoCorrectionStrategy
        self.autocorrection_post_processing = autocorrection_post_processing or self.default_autocorrection_post_processing

    def add_job(self, job : JobInterface):
        self.jobs.insert(0, job)

    def add_job_after(self, job: JobInterface, job_id: str):
        """Add a job after a given job_id"""
        for i, s in enumerate(self.jobs):
            if s.job_id == job_id:
                self.jobs.insert(i + 1, job)
                break

    async def execute(self) -> (Optional[AbstractTask],Optional[BaseAgent]):
        if self.jobs:
            current_job = self.jobs.pop()
            pipeline_response = await self._execute_job(current_job)

            if self.jobs:
                return await self.execute()
        return pipeline_response

    async def _execute_job(self, job : JobInterface):
        # Logic to execute a job and parse the response
        model_response : ChatModelResponse = await self._execute_strategy(strategy_name= job.strategy.STRATEGY_NAME, task = self._task , **job.strategy_kwargs)
        command_name, command_args, assistant_reply_dict = self._parse_response(strategy_name = job.strategy.STRATEGY_NAME,model_response = model_response)

        pipeline_response = job.response_post_process(pipeline = self,
                                                       command_name = command_name,
                                                       command_args = command_args,
                                                       assistant_reply_dict = assistant_reply_dict)
        if job.autocorrection :

            import copy
            new_job =  copy.deepcopy(job)
            new_job.strategy = self.autocorrection_strategy
            new_job.response_post_process = self.autocorrection_post_processing
            new_job.strategy_kwargs = {
                'prompt' : model_response.system_prompt,
                'response' : model_response,
                'corrected_strategy' : job.strategy,
            }
            new_job.autocorrection = False
            self.jobs.append(new_job)
        return pipeline_response

    def _parse_response(self, strategy_name : str, model_response : ChatModelResponse ) :
        strategy_tools = self.get_strategy(strategy_name= strategy_name).get_tools_names()
        for function in model_response.parsed_result :
            if function['command_name'] in strategy_tools :
                return(function['command_name'],
                    function['command_args'],
                function['assistant_reply_dict'])
        return None, None, None

    @classmethod
    def default_post_processing(cls,
                                pipeline : Pipeline ,
                                command_name : str  ,
                                command_args : dict,
                                assistant_reply_dict : Any) :
        pipeline._task.task_text_output = assistant_reply_dict
        pipeline._task.task_context = command_args['note_to_agent']
        return None

    @classmethod
    def default_autocorrection_post_processing(cls,
                                               pipeline : Pipeline ,
                                               command_name : str  , command_args : dict, assistant_reply_dict : Any) :
        return cls.default_post_processing(
                                    pipeline= pipeline,
                                    command_name = command_name,
                                    command_args = command_args,
                                    assistant_reply_dict = assistant_reply_dict
                                    )