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
  isDocPageOutput,
  isDocSearchResultsOutput,
  isErrorOutput,
  isNoResultsOutput,
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

  const docSearchOutput =
    isOutputAvailable && output && isDocSearchResultsOutput(output)
      ? output
      : null;
  const docPageOutput =
    isOutputAvailable && output && isDocPageOutput(output) ? output : null;
  const noResultsOutput =
    isOutputAvailable && output && isNoResultsOutput(output) ? output : null;
  const errorOutput =
    isOutputAvailable && output && isErrorOutput(output) ? output : null;

  const hasExpandableContent =
    isOutputAvailable &&
    ((!!docSearchOutput && docSearchOutput.count > 0) ||
      !!docPageOutput ||
      !!noResultsOutput ||
      !!errorOutput);

  const accordionDescription =
    hasExpandableContent && docSearchOutput
      ? `Found ${docSearchOutput.count} result${docSearchOutput.count === 1 ? "" : "s"} for "${docSearchOutput.query}"`
      : hasExpandableContent && docPageOutput
        ? docPageOutput.path
        : hasExpandableContent && (noResultsOutput || errorOutput)
          ? ((noResultsOutput ?? errorOutput)?.message ?? null)
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
          {docSearchOutput && (
            <div className="grid gap-2">
              {docSearchOutput.results.map((r) => {
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

          {docPageOutput && (
            <div>
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-foreground">
                    {docPageOutput.title}
                  </p>
                  <p className="mt-0.5 truncate text-xs text-muted-foreground">
                    {docPageOutput.path}
                  </p>
                </div>
                <Link
                  href={docPageOutput.doc_url ?? toDocsUrl(docPageOutput.path)}
                  target="_blank"
                  rel="noreferrer"
                  className="shrink-0 text-xs font-medium text-purple-600 hover:text-purple-700"
                >
                  Open
                </Link>
              </div>
              <p className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">
                {truncate(docPageOutput.content, 800)}
              </p>
            </div>
          )}

          {noResultsOutput && (
            <div>
              <p className="text-sm text-foreground">
                {noResultsOutput.message}
              </p>
              {noResultsOutput.suggestions &&
                noResultsOutput.suggestions.length > 0 && (
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-muted-foreground">
                    {noResultsOutput.suggestions.slice(0, 5).map((s) => (
                      <li key={s}>{s}</li>
                    ))}
                  </ul>
                )}
            </div>
          )}

          {errorOutput && (
            <div>
              <p className="text-sm text-foreground">{errorOutput.message}</p>
              {errorOutput.error && (
                <p className="mt-2 text-xs text-muted-foreground">
                  {errorOutput.error}
                </p>
              )}
            </div>
          )}
        </ToolAccordion>
      )}
    </div>
  );
}
