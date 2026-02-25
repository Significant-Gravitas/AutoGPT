"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { WarningDiamondIcon } from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import { AgentSavedCard } from "../../components/AgentSavedCard/AgentSavedCard";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import {
  ContentCardDescription,
  ContentCodeBlock,
  ContentGrid,
  ContentHint,
  ContentMessage,
} from "../../components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  ClarificationQuestionsCard,
  ClarifyingQuestion,
} from "./components/ClarificationQuestionsCard";
import { MiniGame } from "../../components/MiniGame/MiniGame";
import { SuggestedGoalCard } from "./components/SuggestedGoalCard";
import {
  AccordionIcon,
  formatMaybeJson,
  getAnimationText,
  getCreateAgentToolOutput,
  isAgentPreviewOutput,
  isAgentSavedOutput,
  isClarificationNeededOutput,
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

  if (!output) {
    return {
      icon,
      title:
        "Creating agent, this may take a few minutes. Play while you wait.",
      expanded: true,
    };
  }

  if (isAgentSavedOutput(output)) {
    return { icon, title: output.agent_name, expanded: true };
  }
  if (isAgentPreviewOutput(output)) {
    return {
      icon,
      title: output.agent_name,
      description: `${output.node_count} block${output.node_count === 1 ? "" : "s"}`,
    };
  }
  if (isClarificationNeededOutput(output)) {
    const questions = output.questions ?? [];
    return {
      icon,
      title: "Needs clarification",
      description: `${questions.length} question${questions.length === 1 ? "" : "s"}`,
      expanded: true,
    };
  }
  if (isSuggestedGoalOutput(output)) {
    return {
      icon,
      title: "Goal needs refinement",
      expanded: true,
    };
  }
  return {
    icon: (
      <WarningDiamondIcon size={32} weight="light" className="text-red-500" />
    ),
    title: "Error",
    titleClassName: "text-red-500",
  };
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

  // Show accordion for operating state and successful outputs, but not for errors
  // (errors are shown inline so they get replaced when retrying)
  const hasExpandableContent = !isError;

  function handleUseSuggestedGoal(goal: string) {
    onSend(`Please create an agent with this goal: ${goal}`);
  }

  function handleClarificationAnswers(answers: Record<string, string>) {
    const questions =
      output && isClarificationNeededOutput(output)
        ? (output.questions ?? [])
        : [];

    const contextMessage = questions
      .map((q) => {
        const answer = answers[q.keyword] || "";
        return `> ${q.question}\n\n${answer}`;
      })
      .join("\n\n");

    onSend(
      `**Here are my answers:**\n\n${contextMessage}\n\nPlease proceed with creating the agent.`,
    );
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
        <div className="space-y-3 rounded-lg border border-red-200 bg-red-50 p-4">
          <div className="flex items-start gap-2">
            <WarningDiamondIcon
              size={20}
              weight="regular"
              className="mt-0.5 shrink-0 text-red-500"
            />
            <div className="flex-1 space-y-2">
              <Text variant="body-medium" className="text-red-900">
                {output.message ||
                  "Failed to generate the agent. Please try again."}
              </Text>
              {output.error && (
                <details className="-ml-5 pt-2 text-red-700">
                  <summary className="cursor-pointer text-sm font-medium">
                    Technical details
                  </summary>
                  <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-red-100 p-2 text-sm">
                    {formatMaybeJson(output.error)}
                  </pre>
                </details>
              )}
              {output.details && (
                <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-red-100 p-2 text-xs text-red-700">
                  {formatMaybeJson(output.details)}
                </pre>
              )}
            </div>
          </div>
          <div className="flex gap-2 pt-3">
            <Button
              variant="outline"
              size="small"
              onClick={() => onSend("Please try creating the agent again.")}
            >
              Try again
            </Button>
            <Button
              variant="ghost"
              size="small"
              onClick={() => onSend("Can you help me simplify this goal?")}
            >
              Simplify goal
            </Button>
          </div>
        </div>
      )}

      {hasExpandableContent &&
        !(output && isClarificationNeededOutput(output)) &&
        !(output && isAgentSavedOutput(output)) && (
          <ToolAccordion {...getAccordionMeta(output)}>
            {isOperating && (
              <ContentGrid>
                <MiniGame />
                <ContentHint>
                  This could take a few minutes — play while you wait!
                </ContentHint>
              </ContentGrid>
            )}

            {output && isAgentPreviewOutput(output) && (
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

            {output && isSuggestedGoalOutput(output) && (
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

      {output && isClarificationNeededOutput(output) && (
        <ClarificationQuestionsCard
          questions={(output.questions ?? []).map((q) => {
            const item: ClarifyingQuestion = {
              question: q.question,
              keyword: q.keyword,
            };
            const example =
              typeof q.example === "string" && q.example.trim()
                ? q.example.trim()
                : null;
            if (example) item.example = example;
            return item;
          })}
          message={output.message}
          onSubmitAnswers={handleClarificationAnswers}
        />
      )}
    </div>
  );
}
