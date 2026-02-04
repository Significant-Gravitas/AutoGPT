"use client";

import type { ToolUIPart } from "ai";
import Link from "next/link";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import {
  ClarificationQuestionsWidget,
  type ClarifyingQuestion as WidgetClarifyingQuestion,
} from "@/components/contextual/Chat/components/ClarificationQuestionsWidget/ClarificationQuestionsWidget";
import {
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
  StateIcon,
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
  badgeText: string;
  title: string;
  description?: string;
} {
  if (isAgentSavedOutput(output)) {
    return { badgeText: "Edit agent", title: output.agent_name };
  }
  if (isAgentPreviewOutput(output)) {
    return {
      badgeText: "Edit agent",
      title: output.agent_name,
      description: `${output.node_count} block${output.node_count === 1 ? "" : "s"}`,
    };
  }
  if (isClarificationNeededOutput(output)) {
    const questions = output.questions ?? [];
    return {
      badgeText: "Edit agent",
      title: "Needs clarification",
      description: `${questions.length} question${questions.length === 1 ? "" : "s"}`,
    };
  }
  if (
    isOperationStartedOutput(output) ||
    isOperationPendingOutput(output) ||
    isOperationInProgressOutput(output)
  ) {
    return { badgeText: "Edit agent", title: "Editing agent" };
  }
  return { badgeText: "Edit agent", title: "Error" };
}

export function EditAgentTool({ part }: Props) {
  const text = getAnimationText(part);
  const { onSend } = useCopilotChatActions();

  const output = getEditAgentToolOutput(part);
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
    const contextMessage = Object.entries(answers)
      .map(([keyword, answer]) => `${keyword}: ${answer}`)
      .join("\n");

    onSend(
      `I have the answers to your questions:\n\n${contextMessage}\n\nPlease proceed with editing the agent.`,
    );
  }

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <StateIcon state={part.state} />
        <MorphingTextAnimation text={text} />
      </div>

      {hasExpandableContent && output && (
        <ToolAccordion
          {...getAccordionMeta(output)}
          defaultExpanded={isClarificationNeededOutput(output)}
        >
          {(isOperationStartedOutput(output) ||
            isOperationPendingOutput(output)) && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>
              <p className="text-xs text-muted-foreground">
                Operation: {output.operation_id}
              </p>
              <p className="text-xs italic text-muted-foreground">
                Check your library in a few minutes.
              </p>
            </div>
          )}

          {isOperationInProgressOutput(output) && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>
              <p className="text-xs italic text-muted-foreground">
                Please wait for the current operation to finish.
              </p>
            </div>
          )}

          {isAgentSavedOutput(output) && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>
              <div className="flex flex-wrap gap-2">
                <Link
                  href={output.library_agent_link}
                  className="text-xs font-medium text-purple-600 hover:text-purple-700"
                >
                  Open in library
                </Link>
                <Link
                  href={output.agent_page_link}
                  className="text-xs font-medium text-purple-600 hover:text-purple-700"
                >
                  Open in builder
                </Link>
              </div>
              <pre className="whitespace-pre-wrap rounded-2xl border bg-muted/30 p-3 text-xs text-muted-foreground">
                {truncateText(
                  formatMaybeJson({ agent_id: output.agent_id }),
                  800,
                )}
              </pre>
            </div>
          )}

          {isAgentPreviewOutput(output) && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>
              {output.description?.trim() && (
                <p className="text-xs text-muted-foreground">
                  {output.description}
                </p>
              )}
              <pre className="whitespace-pre-wrap rounded-2xl border bg-muted/30 p-3 text-xs text-muted-foreground">
                {truncateText(formatMaybeJson(output.agent_json), 1600)}
              </pre>
            </div>
          )}

          {isClarificationNeededOutput(output) && (
            <ClarificationQuestionsWidget
              questions={(output.questions ?? []).map((q) => {
                const item: WidgetClarifyingQuestion = {
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
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>
              {output.error && (
                <pre className="whitespace-pre-wrap rounded-2xl border bg-muted/30 p-3 text-xs text-muted-foreground">
                  {formatMaybeJson(output.error)}
                </pre>
              )}
              {output.details && (
                <pre className="whitespace-pre-wrap rounded-2xl border bg-muted/30 p-3 text-xs text-muted-foreground">
                  {formatMaybeJson(output.details)}
                </pre>
              )}
            </div>
          )}
        </ToolAccordion>
      )}
    </div>
  );
}
