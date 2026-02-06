"use client";

import { ToolUIPart } from "ai";
import Link from "next/link";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import {
  getAgentHref,
  getAnimationText,
  getFindAgentsOutput,
  getSourceLabelFromToolType,
  isAgentsFoundOutput,
  isErrorOutput,
  ToolIcon,
} from "./helpers";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";

export interface FindAgentsToolPart {
  type: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: FindAgentsToolPart;
}

export function FindAgentsTool({ part }: Props) {
  const text = getAnimationText(part);
  const output = getFindAgentsOutput(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));

  const query =
    typeof part.input === "object" && part.input !== null
      ? String((part.input as { query?: unknown }).query ?? "").trim()
      : "";

  const agentsFoundOutput =
    part.state === "output-available" && output && isAgentsFoundOutput(output)
      ? output
      : null;

  const hasAgents =
    !!agentsFoundOutput &&
    agentsFoundOutput.agents.length > 0 &&
    (typeof agentsFoundOutput.count !== "number" ||
      agentsFoundOutput.count > 0);
  const totalCount = agentsFoundOutput ? agentsFoundOutput.count : 0;
  const { label: sourceLabel, source } = getSourceLabelFromToolType(part.type);
  const scopeText =
    source === "library"
      ? "in your library"
      : source === "marketplace"
        ? "in marketplace"
        : "";
  const accordionDescription = `Found ${totalCount}${scopeText ? ` ${scopeText}` : ""}${
    query ? ` for "${query}"` : ""
  }`;

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolIcon
          toolType={part.type}
          isStreaming={isStreaming}
          isError={isError}
        />
        <MorphingTextAnimation
          text={text}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {hasAgents && agentsFoundOutput && (
        <ToolAccordion
          badgeText={sourceLabel}
          title="Agent results"
          description={accordionDescription}
        >
          <div className="grid gap-2 sm:grid-cols-2">
            {agentsFoundOutput.agents.map((agent) => {
              const href = getAgentHref(agent);
              const agentSource =
                agent.source === "library"
                  ? "Library"
                  : agent.source === "marketplace"
                    ? "Marketplace"
                    : null;
              return (
                <div
                  key={agent.id}
                  className="rounded-2xl border bg-background p-3"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="truncate text-sm font-medium text-foreground">
                          {agent.name}
                        </p>
                        {agentSource && (
                          <span className="shrink-0 rounded-full border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
                            {agentSource}
                          </span>
                        )}
                      </div>
                      <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                        {agent.description}
                      </p>
                    </div>
                    {href && (
                      <Link
                        href={href}
                        className="shrink-0 text-xs font-medium text-purple-600 hover:text-purple-700"
                      >
                        Open
                      </Link>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </ToolAccordion>
      )}
    </div>
  );
}
