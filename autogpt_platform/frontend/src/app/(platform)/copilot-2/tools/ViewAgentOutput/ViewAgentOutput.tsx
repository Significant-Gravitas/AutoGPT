"use client";

import type { ToolUIPart } from "ai";
import Link from "next/link";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  formatMaybeJson,
  getAnimationText,
  getViewAgentOutputToolOutput,
  StateIcon,
  type ViewAgentOutputToolOutput,
} from "./helpers";

export interface ViewAgentOutputToolPart {
  type: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: ViewAgentOutputToolPart;
}

function getAccordionMeta(output: ViewAgentOutputToolOutput): {
  badgeText: string;
  title: string;
  description?: string;
} {
  if (output.type === "agent_output") {
    const status = output.execution?.status;
    return {
      badgeText: "Agent output",
      title: output.agent_name,
      description: status ? `Status: ${status}` : output.message,
    };
  }
  if (output.type === "no_results") {
    return { badgeText: "Agent output", title: "No results" };
  }
  return { badgeText: "Agent output", title: "Error" };
}

export function ViewAgentOutputTool({ part }: Props) {
  const text = getAnimationText(part);

  const output = getViewAgentOutputToolOutput(part);
  const hasExpandableContent =
    part.state === "output-available" &&
    !!output &&
    (output.type === "agent_output" ||
      output.type === "no_results" ||
      output.type === "error");

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <StateIcon state={part.state} />
        <MorphingTextAnimation text={text} />
      </div>

      {hasExpandableContent && output && (
        <ToolAccordion {...getAccordionMeta(output)}>
          {output.type === "agent_output" && (
            <div className="grid gap-2">
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm text-foreground">{output.message}</p>
                {output.library_agent_link && (
                  <Link
                    href={output.library_agent_link}
                    className="shrink-0 text-xs font-medium text-purple-600 hover:text-purple-700"
                  >
                    Open
                  </Link>
                )}
              </div>

              {output.execution ? (
                <div className="grid gap-2">
                  <div className="rounded-2xl border bg-background p-3">
                    <p className="text-xs font-medium text-foreground">
                      Execution
                    </p>
                    <p className="mt-1 truncate text-xs text-muted-foreground">
                      {output.execution.execution_id}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Status: {output.execution.status}
                    </p>
                  </div>

                  {output.execution.inputs_summary && (
                    <div className="rounded-2xl border bg-background p-3">
                      <p className="text-xs font-medium text-foreground">
                        Inputs summary
                      </p>
                      <pre className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">
                        {formatMaybeJson(output.execution.inputs_summary)}
                      </pre>
                    </div>
                  )}

                  {Object.entries(output.execution.outputs ?? {}).map(
                    ([key, items]) => (
                      <div
                        key={key}
                        className="rounded-2xl border bg-background p-3"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <p className="truncate text-xs font-medium text-foreground">
                            {key}
                          </p>
                          <span className="shrink-0 rounded-full border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
                            {items.length} item{items.length === 1 ? "" : "s"}
                          </span>
                        </div>
                        <pre className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">
                          {formatMaybeJson(items.slice(0, 3))}
                        </pre>
                      </div>
                    ),
                  )}
                </div>
              ) : (
                <div className="rounded-2xl border bg-background p-3">
                  <p className="text-sm text-foreground">
                    No execution selected.
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Try asking for a specific run or execution_id.
                  </p>
                </div>
              )}
            </div>
          )}

          {output.type === "no_results" && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>
              {output.suggestions && output.suggestions.length > 0 && (
                <ul className="mt-1 list-disc space-y-1 pl-5 text-xs text-muted-foreground">
                  {output.suggestions.slice(0, 5).map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ul>
              )}
            </div>
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
