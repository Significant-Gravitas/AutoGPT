"""Multi-Agent Debate prompt strategy.

This strategy implements a multi-agent debate approach where multiple sub-agents
propose solutions, debate their merits, and reach a consensus.

Based on research from:
- "Improving Factuality and Reasoning in Language Models through Multiagent Debate"
- Google ADK Multi-Agent Patterns

Key features:
- Multiple sub-agents generate independent proposals
- Agents critique each other's proposals
- Consensus is reached through voting or synthesis
- Improves reasoning through diverse perspectives
"""

from __future__ import annotations

import json
import re
from enum import Enum
from logging import Logger
from typing import Any, Optional

from pydantic import BaseModel, Field

from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.json.parsing import extract_dict_from_json
from forge.llm.prompting import ChatPrompt, LanguageModelClassification
from forge.llm.providers.schema import (
    AssistantChatMessage,
    ChatMessage,
    CompletionModelFunction,
)
from forge.models.action import ActionProposal
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema
from forge.models.utils import ModelWithSummary
from forge.utils.exceptions import InvalidAgentResponseError

from .base import BaseMultiStepPromptStrategy, BasePromptStrategyConfiguration


class DebatePhase(str, Enum):
    """Phases of the multi-agent debate."""

    PROPOSAL = "proposal"  # Agents generate initial proposals
    CRITIQUE = "critique"  # Agents critique each other's proposals
    REVISION = "revision"  # Agents revise based on critiques
    CONSENSUS = "consensus"  # Synthesize final decision
    EXECUTION = "execution"  # Execute the consensus action


class AgentProposal(BaseModel):
    """A proposal from a debate agent."""

    agent_id: str = Field(description="ID of the proposing agent")
    action_name: str = Field(description="Proposed action name")
    action_args: dict[str, Any] = Field(
        default_factory=dict, description="Proposed action arguments"
    )
    reasoning: str = Field(description="Reasoning behind the proposal")
    confidence: float = Field(default=0.5, description="Confidence in proposal (0-1)")


class AgentCritique(BaseModel):
    """A critique of another agent's proposal."""

    critic_id: str = Field(description="ID of the critiquing agent")
    target_agent_id: str = Field(description="ID of the agent being critiqued")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    score: float = Field(default=0.5, description="Score for the proposal (0-1)")


class DebateState(BaseModel):
    """Current state of the debate."""

    proposals: list[AgentProposal] = Field(default_factory=list)
    critiques: list[AgentCritique] = Field(default_factory=list)
    revision_count: int = Field(default=0)
    consensus_reached: bool = Field(default=False)
    winning_proposal: Optional[AgentProposal] = None


class DebateThoughts(ModelWithSummary):
    """Thoughts for debate strategy."""

    observations: str = Field(description="Observations about the debate state")
    debate_summary: str = Field(description="Summary of the debate so far")
    reasoning: str = Field(description="Reasoning for the selected action")
    confidence: float = Field(default=0.5, description="Confidence in decision (0-1)")

    def summary(self) -> str:
        return self.debate_summary


class DebateActionProposal(ActionProposal):
    """Action proposal from debate strategy."""

    thoughts: DebateThoughts  # type: ignore


class MultiAgentDebateConfiguration(BasePromptStrategyConfiguration):
    """Configuration for multi-agent debate strategy."""

    num_debaters: int = UserConfigurable(default=3)
    """Number of debate agents to spawn."""

    num_rounds: int = UserConfigurable(default=2)
    """Number of debate rounds (proposal -> critique -> revision)."""

    consensus_threshold: float = UserConfigurable(default=0.7)
    """Agreement threshold for consensus (0-1)."""

    use_voting: bool = UserConfigurable(default=True)
    """Use voting for consensus vs. synthesis."""

    # Sub-agent configuration
    enable_sub_agents: bool = UserConfigurable(default=True)
    max_sub_agents: int = UserConfigurable(default=10)
    sub_agent_timeout_seconds: int = UserConfigurable(default=180)
    sub_agent_max_cycles: int = UserConfigurable(default=8)

    DEFAULT_PROPOSAL_INSTRUCTION: str = (
        "You are Debater #{agent_num} in a multi-agent debate.\n\n"
        "Task: {task}\n"
        "Available commands: {commands}\n\n"
        "Propose ONE specific action to accomplish this task.\n"
        "Explain your reasoning and why this approach is best.\n\n"
        "Format your response as:\n"
        "ACTION: <command_name>\n"
        "ARGUMENTS: <json arguments>\n"
        "REASONING: <your reasoning>\n"
        "CONFIDENCE: <0.0-1.0>"
    )

    DEFAULT_CRITIQUE_INSTRUCTION: str = (
        "You are a critic evaluating another agent's proposal.\n\n"
        "Task: {task}\n"
        "Proposal being critiqued:\n"
        "- Action: {action}\n"
        "- Arguments: {arguments}\n"
        "- Reasoning: {reasoning}\n\n"
        "Provide a balanced critique:\n"
        "STRENGTHS: <what's good about this proposal>\n"
        "WEAKNESSES: <potential issues or risks>\n"
        "SUGGESTIONS: <how to improve>\n"
        "SCORE: <0.0-1.0>"
    )

    DEFAULT_CONSENSUS_INSTRUCTION: str = (
        "The debate has concluded. Here are the final proposals and their scores:\n\n"
        "{proposals_summary}\n\n"
        "Based on the debate, select the best action to take.\n"
        "You may combine ideas from multiple proposals if beneficial."
    )

    proposal_instruction: str = UserConfigurable(default=DEFAULT_PROPOSAL_INSTRUCTION)
    critique_instruction: str = UserConfigurable(default=DEFAULT_CRITIQUE_INSTRUCTION)
    consensus_instruction: str = UserConfigurable(default=DEFAULT_CONSENSUS_INSTRUCTION)


class MultiAgentDebateStrategy(BaseMultiStepPromptStrategy):
    """Multi-Agent Debate prompt strategy.

    Spawns multiple sub-agents that propose, critique, and debate
    to reach consensus on the best action.
    """

    default_configuration = MultiAgentDebateConfiguration()

    def __init__(
        self,
        configuration: MultiAgentDebateConfiguration,
        logger: Logger,
    ):
        super().__init__(configuration, logger)
        self.config: MultiAgentDebateConfiguration = configuration
        self.response_schema = JSONSchema.from_dict(
            DebateActionProposal.model_json_schema()
        )

        # Debate state
        self.debate_state = DebateState()
        self.phase = DebatePhase.PROPOSAL
        self.current_round = 0
        self._commands_str = ""

    @property
    def llm_classification(self) -> LanguageModelClassification:
        return LanguageModelClassification.SMART_MODEL

    def build_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
        **extras,
    ) -> ChatPrompt:
        """Build prompt based on current debate phase."""
        # Store commands for sub-agents
        self._commands_str = ", ".join(cmd.name for cmd in commands)

        system_prompt = self._build_system_prompt(
            ai_profile, ai_directives, commands, include_os_info
        )

        debate_context = self._build_debate_context()

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'Task: """{task}"""'),
                *messages,
                ChatMessage.user(debate_context),
                ChatMessage.user(self._get_phase_instruction(task)),
            ],
            prefill_response='{\n    "thoughts":',
            functions=commands,
        )

    def _build_system_prompt(
        self,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> str:
        """Build the system prompt."""
        intro = self.generate_intro_prompt(ai_profile)
        body = self.build_body(ai_directives, commands)

        debate_intro = (
            "\n\n## Multi-Agent Debate Strategy\n"
            "You are the coordinator of a multi-agent debate.\n"
            "Multiple agents will propose and critique solutions.\n"
            "Your role is to:\n"
            "1. Orchestrate the debate process\n"
            "2. Synthesize insights from all agents\n"
            "3. Select the best action based on consensus\n"
        )

        response_format = self._build_response_format()

        parts = intro + [body, debate_intro, response_format]
        if include_os_info:
            parts.extend(self.generate_os_info())

        return "\n\n".join(parts)

    def _build_debate_context(self) -> str:
        """Build context about current debate state."""
        context_parts = [
            "## Debate State",
            f"Phase: {self.phase.value}",
            f"Round: {self.current_round + 1}/{self.config.num_rounds}",
            f"Proposals collected: {len(self.debate_state.proposals)}",
            f"Critiques collected: {len(self.debate_state.critiques)}",
        ]

        if self.debate_state.proposals:
            context_parts.append("\n### Current Proposals:")
            for p in self.debate_state.proposals:
                avg_score = self._get_proposal_score(p.agent_id)
                context_parts.append(
                    f"- {p.agent_id}: {p.action_name} (score: {avg_score:.2f})"
                )

        if self.debate_state.consensus_reached:
            context_parts.append("\n✓ Consensus reached!")
            if self.debate_state.winning_proposal:
                wp = self.debate_state.winning_proposal
                context_parts.append(f"Winner: {wp.action_name}")

        return "\n".join(context_parts)

    def _get_proposal_score(self, agent_id: str) -> float:
        """Get average critique score for a proposal."""
        scores = [
            c.score
            for c in self.debate_state.critiques
            if c.target_agent_id == agent_id
        ]
        return sum(scores) / len(scores) if scores else 0.5

    def _get_phase_instruction(self, task: str) -> str:
        """Get instruction for current phase."""
        if self.phase == DebatePhase.PROPOSAL:
            if not self.debate_state.proposals:
                return (
                    "The debate is starting. Sub-agents will now generate proposals. "
                    "Invoke 'finish' with reason 'Starting debate' to begin, "
                    "or take a direct action if you're confident."
                )
            return "Review the proposals and proceed to critique phase."

        elif self.phase == DebatePhase.CRITIQUE:
            return "Sub-agents are critiquing proposals. Proceed to synthesis."

        elif self.phase == DebatePhase.CONSENSUS:
            return self.config.consensus_instruction.format(
                proposals_summary=self._format_proposals_summary()
            )

        else:  # EXECUTION
            return "Execute the consensus action."

    def _format_proposals_summary(self) -> str:
        """Format proposals for consensus instruction."""
        lines = []
        for p in self.debate_state.proposals:
            score = self._get_proposal_score(p.agent_id)
            lines.append(
                f"Proposal from {p.agent_id}:\n"
                f"  Action: {p.action_name}({p.action_args})\n"
                f"  Reasoning: {p.reasoning}\n"
                f"  Score: {score:.2f}"
            )
        return "\n\n".join(lines)

    def _build_response_format(self) -> str:
        """Build response format instruction."""
        response_schema = self.response_schema.model_copy(deep=True)
        if response_schema.properties and "use_tool" in response_schema.properties:
            del response_schema.properties["use_tool"]

        return (
            "## Response Format\n"
            "Respond with a JSON object and invoke a tool.\n"
            f"{response_schema.to_typescript_object_interface('DebateResponse')}"
        )

    async def run_proposal_phase(self, task: str) -> list[AgentProposal]:
        """Run the proposal phase with sub-agents."""
        if not self.can_spawn_sub_agent():
            self.logger.warning("Cannot spawn sub-agents for debate")
            return []

        proposal_tasks = []
        for i in range(self.config.num_debaters):
            sub_task = self.config.proposal_instruction.format(
                agent_num=i + 1,
                task=task,
                commands=self._commands_str,
            )
            proposal_tasks.append(sub_task)

        try:
            results = await self.run_parallel(
                proposal_tasks,
                strategy="one_shot",
                max_cycles=self.config.sub_agent_max_cycles,
            )

            proposals = []
            for i, result in enumerate(results):
                if result:
                    proposal = self._parse_proposal(f"debater-{i + 1}", str(result))
                    if proposal:
                        proposals.append(proposal)

            self.debate_state.proposals = proposals
            self.phase = DebatePhase.CRITIQUE
            return proposals

        except Exception as e:
            self.logger.error(f"Proposal phase failed: {e}")
            return []

    def _parse_proposal(self, agent_id: str, result: str) -> Optional[AgentProposal]:
        """Parse a proposal from sub-agent output."""
        try:
            # Try to extract structured data
            action_match = re.search(r"ACTION:\s*(\w+)", result, re.IGNORECASE)
            args_match = re.search(r"ARGUMENTS:\s*(\{.*?\})", result, re.DOTALL)
            reasoning_match = re.search(
                r"REASONING:\s*(.+?)(?=CONFIDENCE:|$)",
                result,
                re.DOTALL | re.IGNORECASE,
            )
            confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", result)

            if action_match:
                action_name = action_match.group(1)
                action_args = {}
                if args_match:
                    try:
                        action_args = json.loads(args_match.group(1))
                    except json.JSONDecodeError:
                        pass

                reasoning = (
                    reasoning_match.group(1).strip()
                    if reasoning_match
                    else result[:200]
                )
                confidence = (
                    float(confidence_match.group(1)) if confidence_match else 0.5
                )

                return AgentProposal(
                    agent_id=agent_id,
                    action_name=action_name,
                    action_args=action_args,
                    reasoning=reasoning,
                    confidence=min(confidence, 1.0),
                )

            # Fallback: try to extract any useful info
            return AgentProposal(
                agent_id=agent_id,
                action_name="unknown",
                action_args={},
                reasoning=result[:300],
                confidence=0.3,
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse proposal: {e}")
            return None

    async def run_critique_phase(self, task: str) -> list[AgentCritique]:
        """Run the critique phase with sub-agents."""
        if not self.can_spawn_sub_agent() or not self.debate_state.proposals:
            return []

        critique_tasks = []
        for i, proposal in enumerate(self.debate_state.proposals):
            # Each other debater critiques this proposal
            for j in range(self.config.num_debaters):
                if j == i:
                    continue  # Don't critique own proposal

                sub_task = self.config.critique_instruction.format(
                    task=task,
                    action=proposal.action_name,
                    arguments=json.dumps(proposal.action_args),
                    reasoning=proposal.reasoning,
                )
                critique_tasks.append((f"critic-{j + 1}", proposal.agent_id, sub_task))

        try:
            # Run critiques (limit parallelism)
            critiques = []
            for critic_id, target_id, sub_task in critique_tasks:
                result = await self.spawn_and_run(
                    sub_task,
                    strategy="one_shot",
                    max_cycles=5,
                )
                if result:
                    critique = self._parse_critique(critic_id, target_id, str(result))
                    if critique:
                        critiques.append(critique)

            self.debate_state.critiques = critiques
            self.current_round += 1

            if self.current_round >= self.config.num_rounds:
                self.phase = DebatePhase.CONSENSUS
            else:
                self.phase = DebatePhase.PROPOSAL  # Another round

            return critiques

        except Exception as e:
            self.logger.error(f"Critique phase failed: {e}")
            return []

    def _parse_critique(
        self, critic_id: str, target_id: str, result: str
    ) -> Optional[AgentCritique]:
        """Parse a critique from sub-agent output."""
        try:
            strengths = re.findall(
                r"STRENGTHS?:\s*(.+?)(?=WEAKNESSES?:|$)",
                result,
                re.DOTALL | re.IGNORECASE,
            )
            weaknesses = re.findall(
                r"WEAKNESSES?:\s*(.+?)(?=SUGGESTIONS?:|$)",
                result,
                re.DOTALL | re.IGNORECASE,
            )
            suggestions = re.findall(
                r"SUGGESTIONS?:\s*(.+?)(?=SCORE:|$)", result, re.DOTALL | re.IGNORECASE
            )
            score_match = re.search(r"SCORE:\s*([\d.]+)", result)

            return AgentCritique(
                critic_id=critic_id,
                target_agent_id=target_id,
                strengths=[s.strip() for s in strengths] if strengths else [],
                weaknesses=[w.strip() for w in weaknesses] if weaknesses else [],
                suggestions=[s.strip() for s in suggestions] if suggestions else [],
                score=float(score_match.group(1)) if score_match else 0.5,
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse critique: {e}")
            return None

    def determine_consensus(self) -> Optional[AgentProposal]:
        """Determine consensus from proposals and critiques."""
        if not self.debate_state.proposals:
            return None

        # Score proposals by average critique score
        scored_proposals = []
        for proposal in self.debate_state.proposals:
            score = self._get_proposal_score(proposal.agent_id)
            # Also factor in original confidence
            combined_score = 0.7 * score + 0.3 * proposal.confidence
            scored_proposals.append((proposal, combined_score))

        # Sort by score
        scored_proposals.sort(key=lambda x: x[1], reverse=True)

        # Check if top proposal meets consensus threshold
        if scored_proposals:
            best_proposal, best_score = scored_proposals[0]
            if best_score >= self.config.consensus_threshold:
                self.debate_state.consensus_reached = True
                self.debate_state.winning_proposal = best_proposal
                self.phase = DebatePhase.EXECUTION
                return best_proposal

        # If no clear winner, return highest scored anyway
        if scored_proposals:
            self.debate_state.winning_proposal = scored_proposals[0][0]
            return scored_proposals[0][0]

        return None

    def parse_response_content(
        self,
        response: AssistantChatMessage,
    ) -> DebateActionProposal:
        """Parse the LLM response into a debate action proposal."""
        if not response.content:
            raise InvalidAgentResponseError("Assistant response has no text content")

        self.logger.debug(f"LLM response content:\n{response.content[:500]}")

        assistant_reply_dict = extract_dict_from_json(response.content)

        if not response.tool_calls:
            raise InvalidAgentResponseError("Assistant did not use a tool")

        assistant_reply_dict["use_tool"] = response.tool_calls[0].function

        parsed_response = DebateActionProposal.model_validate(assistant_reply_dict)
        parsed_response.raw_message = response.model_copy()

        return parsed_response

    def record_execution_result(
        self, variable_name: str, result: str, error: Optional[str] = None
    ) -> None:
        """Record execution result."""
        # Reset for next decision if needed
        if self.phase == DebatePhase.EXECUTION:
            self.debate_state = DebateState()
            self.phase = DebatePhase.PROPOSAL
            self.current_round = 0
