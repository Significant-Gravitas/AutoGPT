"use client";

import type { ToolUIPart } from "ai";
import { useMemo } from "react";

import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import {
  ContentCard,
  ContentCardDescription,
  ContentCardHeader,
  ContentCardSubtitle,
  ContentCardTitle,
  ContentGrid,
  ContentLink,
  ContentMessage,
  ContentSuggestionsList,
} from "../../components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  AccordionIcon,
  getAnimationText,
  getDocsToolOutput,
  getDocsToolTitle,
  getToolLabel,
  isDocPageOutput,
  isDocSearchResultsOutput,
  isErrorOutput,
  isNoResultsOutput,
  toDocsUrl,
  ToolIcon,
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
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));

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

      {hasExpandableContent && normalized && (
        <ToolAccordion
          icon={<AccordionIcon toolType={part.type} />}
          title={normalized.title}
          description={accordionDescription}
        >
          {docSearchOutput && (
            <ContentGrid>
              {docSearchOutput.results.map((r) => {
                const href = r.doc_url ?? toDocsUrl(r.path);
                return (
                  <ContentCard key={r.path}>
                    <ContentCardHeader
                      action={<ContentLink href={href}>Open</ContentLink>}
                    >
                      <ContentCardTitle>{r.title}</ContentCardTitle>
                      <ContentCardSubtitle>
                        {r.path}
                        {r.section ? ` • ${r.section}` : ""}
                      </ContentCardSubtitle>
                      <ContentCardDescription>
                        {truncate(r.snippet, 240)}
                      </ContentCardDescription>
                    </ContentCardHeader>
                  </ContentCard>
                );
              })}
            </ContentGrid>
          )}

          {docPageOutput && (
            <div>
              <ContentCardHeader
                action={
                  <ContentLink
                    href={
                      docPageOutput.doc_url ?? toDocsUrl(docPageOutput.path)
                    }
                  >
                    Open
                  </ContentLink>
                }
              >
                <ContentCardTitle>{docPageOutput.title}</ContentCardTitle>
                <ContentCardSubtitle>{docPageOutput.path}</ContentCardSubtitle>
              </ContentCardHeader>
              <ContentCardDescription className="whitespace-pre-wrap">
                {truncate(docPageOutput.content, 800)}
              </ContentCardDescription>
            </div>
          )}

          {noResultsOutput && (
            <div>
              <ContentMessage>{noResultsOutput.message}</ContentMessage>
              {noResultsOutput.suggestions &&
                noResultsOutput.suggestions.length > 0 && (
                  <ContentSuggestionsList items={noResultsOutput.suggestions} />
                )}
            </div>
          )}

          {errorOutput && (
            <div>
              <ContentMessage>{errorOutput.message}</ContentMessage>
              {errorOutput.error && (
                <ContentCardDescription>
                  {errorOutput.error}
                </ContentCardDescription>
              )}
            </div>
          )}
        </ToolAccordion>
      )}
    </div>
  );
}
