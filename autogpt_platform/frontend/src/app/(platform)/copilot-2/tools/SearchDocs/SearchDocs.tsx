"use client";

import type { ToolUIPart } from "ai";
import Link from "next/link";
import { useMemo } from "react";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  getDocsToolOutput,
  getDocsToolTitle,
  getToolLabel,
  getAnimationText,
  StateIcon,
  toDocsUrl,
  type DocsToolType,
} from "./helpers";

export interface DocsToolPart {
  type: DocsToolType;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: DocsToolPart;
}

function truncate(text: string, maxChars: number): string {
  const trimmed = text.trim();
  if (trimmed.length <= maxChars) return trimmed;
  return `${trimmed.slice(0, maxChars).trimEnd()}…`;
}

export function SearchDocsTool({ part }: Props) {
  const output = getDocsToolOutput(part);
  const text = getAnimationText(part);

  const normalized = useMemo(() => {
    if (!output) return null;
    const title = getDocsToolTitle(part.type, output);
    const label = getToolLabel(part.type);
    return { title, label };
  }, [output, part.type]);

  const isOutputAvailable = part.state === "output-available" && !!output;

  const hasExpandableContent =
    isOutputAvailable &&
    ((output.type === "doc_search_results" && output.count > 0) ||
      output.type === "doc_page" ||
      output.type === "no_results" ||
      output.type === "error");

  const accordionDescription =
    hasExpandableContent && output
      ? output.type === "doc_search_results"
        ? `Found ${output.count} result${output.count === 1 ? "" : "s"} for "${output.query}"`
        : output.type === "doc_page"
          ? output.path
          : output.message
      : null;

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <StateIcon state={part.state} />
        <MorphingTextAnimation text={text} />
      </div>

      {hasExpandableContent && normalized && (
        <ToolAccordion
          badgeText={normalized.label}
          title={normalized.title}
          description={accordionDescription}
        >
          {output.type === "doc_search_results" && (
            <div className="grid gap-2">
              {output.results.map((r) => {
                const href = r.doc_url ?? toDocsUrl(r.path);
                return (
                  <div
                    key={r.path}
                    className="rounded-2xl border bg-background p-3"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-foreground">
                          {r.title}
                        </p>
                        <p className="mt-0.5 truncate text-xs text-muted-foreground">
                          {r.path}
                          {r.section ? ` • ${r.section}` : ""}
                        </p>
                        <p className="mt-2 text-xs text-muted-foreground">
                          {truncate(r.snippet, 240)}
                        </p>
                      </div>
                      <Link
                        href={href}
                        target="_blank"
                        rel="noreferrer"
                        className="shrink-0 text-xs font-medium text-purple-600 hover:text-purple-700"
                      >
                        Open
                      </Link>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {output.type === "doc_page" && (
            <div>
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-foreground">
                    {output.title}
                  </p>
                  <p className="mt-0.5 truncate text-xs text-muted-foreground">
                    {output.path}
                  </p>
                </div>
                <Link
                  href={output.doc_url ?? toDocsUrl(output.path)}
                  target="_blank"
                  rel="noreferrer"
                  className="shrink-0 text-xs font-medium text-purple-600 hover:text-purple-700"
                >
                  Open
                </Link>
              </div>
              <p className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">
                {truncate(output.content, 800)}
              </p>
            </div>
          )}

          {output.type === "no_results" && (
            <div>
              <p className="text-sm text-foreground">{output.message}</p>
              {output.suggestions && output.suggestions.length > 0 && (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-muted-foreground">
                  {output.suggestions.slice(0, 5).map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {output.type === "error" && (
            <div>
              <p className="text-sm text-foreground">{output.message}</p>
              {output.error && (
                <p className="mt-2 text-xs text-muted-foreground">
                  {output.error}
                </p>
              )}
            </div>
          )}
        </ToolAccordion>
      )}
    </div>
  );
}
