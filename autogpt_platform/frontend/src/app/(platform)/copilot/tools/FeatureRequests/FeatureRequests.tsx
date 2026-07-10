"use client";

import type { ToolUIPart } from "ai";

import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import {
  ContentBadge,
  ContentCard,
  ContentCardDescription,
  ContentCardHeader,
  ContentCardTitle,
  ContentGrid,
  ContentMessage,
  ContentSuggestionsList,
} from "../../components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  getAnimationText,
  getFeatureRequestOutput,
  isCreatedOutput,
  isErrorOutput,
  isNoResultsOutput,
  isSearchResultsOutput,
  ToolIcon,
  type FeatureRequestToolType,
} from "./helpers";

export interface FeatureRequestToolPart {
  type: FeatureRequestToolType;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: FeatureRequestToolPart;
}

function truncate(text: string, maxChars: number): string {
  const trimmed = text.trim();
  if (trimmed.length <= maxChars) return trimmed;
  return `${trimmed.slice(0, maxChars).trimEnd()}…`;
}

export function SearchFeatureRequestsTool({ part }: Props) {
  const output = getFeatureRequestOutput(part);
  const text = getAnimationText(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));

  const isOutputAvailable = part.state === "output-available" && !!output;

  const searchOutput =
    isOutputAvailable && output && isSearchResultsOutput(output)
      ? output
      : null;
  const noResultsOutput =
    isOutputAvailable && output && isNoResultsOutput(output) ? output : null;
  const errorOutput =
    isOutputAvailable && output && isErrorOutput(output) ? output : null;

  const hasExpandableContent =
    isOutputAvailable &&
    ((!!searchOutput && searchOutput.count > 0) ||
      !!noResultsOutput ||
      !!errorOutput);

  // With output, the accordion header IS the tool row — a single compact
  // line (icon + summary + caret), matching the bash-style tools.
  if (hasExpandableContent) {
    return (
      <div className="py-1">
        <ToolAccordion
          variant="compact"
          icon={<ToolIcon toolType={part.type} isError={isError} />}
          title={text}
          titleClassName={isError ? "text-red-500" : undefined}
        >
          {searchOutput && (
            <ContentGrid>
              {searchOutput.results.map((r) => (
                <ContentCard key={r.id}>
                  <ContentCardHeader>
                    <ContentCardTitle>{r.title}</ContentCardTitle>
                  </ContentCardHeader>
                  {r.description && (
                    <ContentCardDescription>
                      {truncate(r.description, 200)}
                    </ContentCardDescription>
                  )}
                </ContentCard>
              ))}
            </ContentGrid>
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
      </div>
    );
  }

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
    </div>
  );
}

export function CreateFeatureRequestTool({ part }: Props) {
  const output = getFeatureRequestOutput(part);
  const text = getAnimationText(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));

  const isOutputAvailable = part.state === "output-available" && !!output;

  const createdOutput =
    isOutputAvailable && output && isCreatedOutput(output) ? output : null;
  const errorOutput =
    isOutputAvailable && output && isErrorOutput(output) ? output : null;

  const hasExpandableContent =
    isOutputAvailable && (!!createdOutput || !!errorOutput);

  // With output, the accordion header IS the tool row — a single compact
  // line (icon + summary + caret), matching the bash-style tools.
  if (hasExpandableContent) {
    return (
      <div className="py-1">
        <ToolAccordion
          variant="compact"
          icon={<ToolIcon toolType={part.type} isError={isError} />}
          title={text}
          titleClassName={isError ? "text-red-500" : undefined}
        >
          {createdOutput && (
            <ContentCard>
              <ContentCardHeader>
                <ContentCardTitle>{createdOutput.issue_title}</ContentCardTitle>
              </ContentCardHeader>
              <div className="mt-2 flex items-center gap-2">
                <ContentBadge>
                  {createdOutput.is_new_issue ? "New" : "Existing"}
                </ContentBadge>
              </div>
              <ContentMessage>{createdOutput.message}</ContentMessage>
            </ContentCard>
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
      </div>
    );
  }

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
    </div>
  );
}
