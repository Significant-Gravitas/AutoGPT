"use client";

import type { ToolUIPart } from "ai";
import Link from "next/link";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { ClarificationQuestionsWidget } from "@/components/contextual/Chat/components/ClarificationQuestionsWidget/ClarificationQuestionsWidget";
import {
  formatMaybeJson,
  getAnimationText,
  getEditAgentToolOutput,
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
  if (output.type === "agent_saved") {
    return { badgeText: "Edit agent", title: output.agent_name };
  }
  if (output.type === "agent_preview") {
    return {
      badgeText: "Edit agent",
      title: output.agent_name,
      description: `${output.node_count} block${output.node_count === 1 ? "" : "s"}`,
    };
  }
  if (output.type === "clarification_needed") {
    return {
      badgeText: "Edit agent",
      title: "Needs clarification",
      description: `${output.questions.length} question${output.questions.length === 1 ? "" : "s"}`,
    };
  }
  if (
    output.type === "operation_started" ||
    output.type === "operation_pending" ||
    output.type === "operation_in_progress"
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
    (output.type === "operation_started" ||
      output.type === "operation_pending" ||
      output.type === "operation_in_progress" ||
      output.type === "agent_preview" ||
      output.type === "agent_saved" ||
      output.type === "clarification_needed" ||
      output.type === "error");

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
          defaultExpanded={output.type === "clarification_needed"}
        >
          {(output.type === "operation_started" ||
            output.type === "operation_pending") && (
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

          {output.type === "operation_in_progress" && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>
              <p className="text-xs italic text-muted-foreground">
                Please wait for the current operation to finish.
              </p>
            </div>
          )}

          {output.type === "agent_saved" && (
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

          {output.type === "agent_preview" && (
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

          {output.type === "clarification_needed" && (
            <ClarificationQuestionsWidget
              questions={output.questions}
              message={output.message}
              onSubmitAnswers={handleClarificationAnswers}
            />
          )}

          {output.type === "error" && (
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
