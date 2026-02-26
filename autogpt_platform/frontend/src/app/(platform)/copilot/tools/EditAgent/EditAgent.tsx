"use client";

import type { ToolUIPart } from "ai";
import { AgentSavedCard } from "../../components/AgentSavedCard/AgentSavedCard";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { ToolErrorCard } from "../../components/ToolErrorCard/ToolErrorCard";
import { MiniGame } from "../../components/MiniGame/MiniGame";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import {
  ContentCardDescription,
  ContentCodeBlock,
  ContentGrid,
  ContentHint,
  ContentMessage,
} from "../../components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { ClarificationQuestionsCard } from "../CreateAgent/components/ClarificationQuestionsCard";
import { normalizeClarifyingQuestions } from "../CreateAgent/helpers";
import {
  AccordionIcon,
  formatMaybeJson,
  getAnimationText,
  getEditAgentToolOutput,
  isAgentPreviewOutput,
  isAgentSavedOutput,
  isClarificationNeededOutput,
  isErrorOutput,
  ToolIcon,
  truncateText,
  type EditAgentToolOutput,
} from "./helpers";

export interface EditAgentToolPart {
  type: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: EditAgentToolPart;
}

function getAccordionMeta(output: EditAgentToolOutput | null): {
  icon: React.ReactNode;
  title: string;
  titleClassName?: string;
  description?: string;
  expanded?: boolean;
} {
  const icon = <AccordionIcon />;

  if (!output) {
    return {
      icon,
      title: "Editing agent, this may take a few minutes. Play while you wait.",
      expanded: true,
    };
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
    };
  }
  return { icon, title: "" };
}

export function EditAgentTool({ part }: Props) {
  const text = getAnimationText(part);
  const { onSend } = useCopilotChatActions();
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  const output = getEditAgentToolOutput(part);
  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));

  const isOperating = !output;

  // Show accordion for operating state and successful outputs, but not for errors
  // (errors are shown inline so they get replaced when retrying)
  const hasExpandableContent = !isError;

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
      `**Here are my answers:**\n\n${contextMessage}\n\nPlease proceed with editing the agent.`,
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
        <ToolErrorCard
          message={output.message}
          fallbackMessage="Failed to edit the agent. Please try again."
          error={output.error ? formatMaybeJson(output.error) : undefined}
          details={output.details ? formatMaybeJson(output.details) : undefined}
          actions={[
            {
              label: "Try again",
              onClick: () => onSend("Please try editing the agent again."),
            },
          ]}
        />
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
          </ToolAccordion>
        )}

      {output && isAgentSavedOutput(output) && (
        <AgentSavedCard
          agentName={output.agent_name}
          message="has been updated!"
          libraryAgentLink={output.library_agent_link}
          agentPageLink={output.agent_page_link}
        />
      )}

      {output && isClarificationNeededOutput(output) && (
        <ClarificationQuestionsCard
          questions={normalizeClarifyingQuestions(output.questions ?? [])}
          message={output.message}
          onSubmitAnswers={handleClarificationAnswers}
        />
      )}
    </div>
  );
}
