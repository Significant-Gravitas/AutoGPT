"use client";

import { WarningDiamondIcon } from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { OrbitLoader } from "../../components/OrbitLoader/OrbitLoader";
import { ProgressBar } from "../../components/ProgressBar/ProgressBar";
import {
  ContentCardDescription,
  ContentCodeBlock,
  ContentGrid,
  ContentHint,
  ContentLink,
  ContentMessage,
} from "../../components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { useAsymptoticProgress } from "../../hooks/useAsymptoticProgress";
import {
  ClarificationQuestionsCard,
  ClarifyingQuestion,
} from "../CreateAgent/components/ClarificationQuestionsCard";
import {
  AccordionIcon,
  formatMaybeJson,
  getAnimationText,
  getEditAgentToolOutput,
  isAgentPreviewOutput,
  isAgentSavedOutput,
  isClarificationNeededOutput,
  isErrorOutput,
  isOperationInProgressOutput,
  isOperationPendingOutput,
  isOperationStartedOutput,
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

function getAccordionMeta(output: EditAgentToolOutput): {
  icon: React.ReactNode;
  title: string;
  titleClassName?: string;
  description?: string;
} {
  const icon = <AccordionIcon />;

  if (isAgentSavedOutput(output)) {
    return { icon, title: output.agent_name };
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
  if (
    isOperationStartedOutput(output) ||
    isOperationPendingOutput(output) ||
    isOperationInProgressOutput(output)
  ) {
    return { icon: <OrbitLoader size={32} />, title: "Editing agent" };
  }
  return {
    icon: (
      <WarningDiamondIcon size={32} weight="light" className="text-red-500" />
    ),
    title: "Error",
    titleClassName: "text-red-500",
  };
}

export function EditAgentTool({ part }: Props) {
  const text = getAnimationText(part);
  const { onSend } = useCopilotChatActions();
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  const output = getEditAgentToolOutput(part);
  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));
  const isOperating =
    !!output &&
    (isOperationStartedOutput(output) ||
      isOperationPendingOutput(output) ||
      isOperationInProgressOutput(output));
  const progress = useAsymptoticProgress(isOperating);
  const hasExpandableContent =
    part.state === "output-available" &&
    !!output &&
    (isOperationStartedOutput(output) ||
      isOperationPendingOutput(output) ||
      isOperationInProgressOutput(output) ||
      isAgentPreviewOutput(output) ||
      isAgentSavedOutput(output) ||
      isClarificationNeededOutput(output) ||
      isErrorOutput(output));

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
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolIcon isStreaming={isStreaming} isError={isError} />
        <MorphingTextAnimation
          text={text}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {hasExpandableContent && output && (
        <ToolAccordion {...getAccordionMeta(output)}>
          {isOperating && (
            <ContentGrid>
              <ProgressBar value={progress} className="max-w-[280px]" />
              <ContentHint>
                This could take a few minutes, grab a coffee ☕
              </ContentHint>
            </ContentGrid>
          )}

          {isAgentSavedOutput(output) && (
            <ContentGrid>
              <ContentMessage>{output.message}</ContentMessage>
              <div className="flex flex-wrap gap-2">
                <ContentLink href={output.library_agent_link}>
                  Open in library
                </ContentLink>
                <ContentLink href={output.agent_page_link}>
                  Open in builder
                </ContentLink>
              </div>
              <ContentCodeBlock>
                {truncateText(
                  formatMaybeJson({ agent_id: output.agent_id }),
                  800,
                )}
              </ContentCodeBlock>
            </ContentGrid>
          )}

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

          {isClarificationNeededOutput(output) && (
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

          {isErrorOutput(output) && (
            <ContentGrid>
              <ContentMessage>{output.message}</ContentMessage>
              {output.error && (
                <ContentCodeBlock>
                  {formatMaybeJson(output.error)}
                </ContentCodeBlock>
              )}
              {output.details && (
                <ContentCodeBlock>
                  {formatMaybeJson(output.details)}
                </ContentCodeBlock>
              )}
            </ContentGrid>
          )}
        </ToolAccordion>
      )}
    </div>
  );
}
