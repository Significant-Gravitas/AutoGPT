

from autogpt.agents.components import BuildPrompt, Component, ResponseHandler


class OneShotComponent(Component, BuildPrompt, ResponseHandler):
    def get_prompt(self, result: BuildPrompt.Result) -> None:
        ai_directives = self.directives.copy(deep=True)
        ai_directives.resources += scratchpad.resources
        ai_directives.constraints += scratchpad.constraints
        ai_directives.best_practices += scratchpad.best_practices
        extra_commands += list(scratchpad.commands.values())

        prompt = self.prompt_strategy.build_prompt(
            task=self.state.task,
            ai_profile=self.ai_profile,
            ai_directives=ai_directives,
            commands=get_openai_command_specs(
                self.command_registry.list_available_commands(self)
            )
            + extra_commands,
            event_history=self.event_history,
            max_prompt_tokens=self.send_token_limit,
            count_tokens=lambda x: self.llm_provider.count_tokens(x, self.llm.name),
            count_message_tokens=lambda x: self.llm_provider.count_message_tokens(
                x, self.llm.name
            ),
            extra_messages=extra_messages,
            **extras,
        )

        return prompt
    
    def parse_and_process_response(
        self, llm_response: AssistantChatMessage
    ) -> Agent.ThoughtProcessOutput:

        (
            command_name,
            arguments,
            assistant_reply_dict,
        ) = self.prompt_strategy.parse_response_content(llm_response)