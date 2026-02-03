"use client";

import type { ToolUIPart } from "ai";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  formatMaybeJson,
  getAnimationText,
  getRunBlockToolOutput,
  StateIcon,
  type RunBlockToolOutput,
} from "./helpers";

export interface RunBlockToolPart {
  type: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: RunBlockToolPart;
}

function getAccordionMeta(output: RunBlockToolOutput): {
  badgeText: string;
  title: string;
  description?: string;
} {
  if (output.type === "block_output") {
    const keys = Object.keys(output.outputs ?? {});
    return {
      badgeText: "Run block",
      title: output.block_name,
      description:
        keys.length > 0
          ? `${keys.length} output key${keys.length === 1 ? "" : "s"}`
          : output.message,
    };
  }

  if (output.type === "setup_requirements") {
    const missingCredsCount = Object.keys(
      output.setup_info.user_readiness.missing_credentials ?? {},
    ).length;
    return {
      badgeText: "Run block",
      title: output.setup_info.agent_name,
      description:
        missingCredsCount > 0
          ? `Missing ${missingCredsCount} credential${missingCredsCount === 1 ? "" : "s"}`
          : output.message,
    };
  }

  return { badgeText: "Run block", title: "Error" };
}

export function RunBlockTool({ part }: Props) {
  const text = getAnimationText(part);

  const output = getRunBlockToolOutput(part);
  const hasExpandableContent =
    part.state === "output-available" &&
    !!output &&
    (output.type === "block_output" ||
      output.type === "setup_requirements" ||
      output.type === "error");

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <StateIcon state={part.state} />
        <MorphingTextAnimation text={text} />
      </div>

      {hasExpandableContent && output && (
        <ToolAccordion {...getAccordionMeta(output)}>
          {output.type === "block_output" && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>

              {Object.entries(output.outputs ?? {}).map(([key, items]) => (
                <div key={key} className="rounded-2xl border bg-background p-3">
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
              ))}
            </div>
          )}

          {output.type === "setup_requirements" && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>

              {Object.keys(
                output.setup_info.user_readiness.missing_credentials ?? {},
              ).length > 0 && (
                <div className="rounded-2xl border bg-background p-3">
                  <p className="text-xs font-medium text-foreground">
                    Missing credentials
                  </p>
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-muted-foreground">
                    {Object.entries(
                      output.setup_info.user_readiness.missing_credentials ??
                        {},
                    ).map(([field, cred]) => (
                      <li key={field}>
                        {cred.title} ({cred.provider})
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {output.setup_info.requirements.inputs?.length > 0 && (
                <div className="rounded-2xl border bg-background p-3">
                  <p className="text-xs font-medium text-foreground">
                    Expected inputs
                  </p>
                  <div className="mt-2 grid gap-2">
                    {output.setup_info.requirements.inputs.map((input) => (
                      <div key={input.name} className="rounded-xl border p-2">
                        <div className="flex items-center justify-between gap-2">
                          <p className="truncate text-xs font-medium text-foreground">
                            {input.title}
                          </p>
                          <span className="shrink-0 rounded-full border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
                            {input.required ? "Required" : "Optional"}
                          </span>
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {input.name} • {input.type}
                          {input.description ? ` • ${input.description}` : ""}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
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
