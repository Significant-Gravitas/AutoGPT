"use client";

import type { ToolUIPart } from "ai";
import { AgentSavedCard } from "../../components/AgentSavedCard/AgentSavedCard";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { ToolErrorCard } from "../../components/ToolErrorCard/ToolErrorCard";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import {
  ContentCardDescription,
  ContentCodeBlock,
  ContentGrid,
  ContentMessage,
} from "../../components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { SuggestedGoalCard } from "./components/SuggestedGoalCard";
import {
  AccordionIcon,
  formatMaybeJson,
  getAnimationText,
  getCreateAgentToolOutput,
  isAgentPreviewOutput,
  isAgentSavedOutput,
  isErrorOutput,
  isSuggestedGoalOutput,
  ToolIcon,
  truncateText,
  type CreateAgentToolOutput,
} from "./helpers";

export interface CreateAgentToolPart {
  type: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: CreateAgentToolPart;
}

function getAccordionMeta(output: CreateAgentToolOutput | null) {
  const icon = <AccordionIcon />;

  if (output && isAgentPreviewOutput(output)) {
    return {
      icon,
      title: output.agent_name,
      description: `${output.node_count} block${output.node_count === 1 ? "" : "s"}`,
    };
  }
  if (output && isSuggestedGoalOutput(output)) {
    return {
      icon,
      title: "Goal needs refinement",
      expanded: true,
    };
  }
  return { icon, title: "" };
}

export function CreateAgentTool({ part }: Props) {
  const text = getAnimationText(part);
  const { onSend } = useCopilotChatActions();

  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  const output = getCreateAgentToolOutput(part);

  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));

  const isOperating = !output;

  function handleUseSuggestedGoal(goal: string) {
    onSend(`Please create an agent with this goal: ${goal}`);
  }

  return (
    <div className="py-2">
      {isOperating && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <ToolIcon isStreaming={isStreaming} isError={isError} />
          <MorphingTextAnimation
            text={text}
            className={isError ? "text-red-500" : undefined}
          />
        </div>
      )}

      {isError && output && isErrorOutput(output) && (
        <ToolErrorCard
          message={output.message}
          fallbackMessage="Failed to generate the agent. Please try again."
          error={output.error ? formatMaybeJson(output.error) : undefined}
          details={output.details ? formatMaybeJson(output.details) : undefined}
          actions={[
            {
              label: "Try again",
              onClick: () => onSend("Please try creating the agent again."),
            },
            {
              label: "Simplify goal",
              variant: "ghost",
              onClick: () => onSend("Can you help me simplify this goal?"),
            },
          ]}
        />
      )}

      {output &&
        !isError &&
        (isAgentPreviewOutput(output) || isSuggestedGoalOutput(output)) && (
          <ToolAccordion {...getAccordionMeta(output)}>
            {isAgentPreviewOutput(output) && (
              <ContentGrid>
                <ContentMessage>{output.message}</ContentMessage>
                {output.description?.trim() && (
                  <ContentCardDescription>
                    {output.description}
                  </ContentCardDescription>
                )}
                <ContentCodeBlock>
                  {truncateText(formatMaybeJson(output.agent_json), 1600)}
                </ContentCodeBlock>
              </ContentGrid>
            )}

            {isSuggestedGoalOutput(output) && (
              <SuggestedGoalCard
                message={output.message}
                suggestedGoal={output.suggested_goal}
                reason={output.reason}
                goalType={output.goal_type ?? "vague"}
                onUseSuggestedGoal={handleUseSuggestedGoal}
              />
            )}
          </ToolAccordion>
        )}

      {output && isAgentSavedOutput(output) && (
        <AgentSavedCard
          agentName={output.agent_name}
          message="has been saved to your library!"
          libraryAgentLink={output.library_agent_link}
          agentPageLink={output.agent_page_link}
        />
      )}
    </div>
  );
}
