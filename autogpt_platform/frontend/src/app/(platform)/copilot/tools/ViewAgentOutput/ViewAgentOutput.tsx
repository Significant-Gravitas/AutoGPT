"use client";

import type { ToolUIPart } from "ai";
import React from "react";
import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import {
  globalRegistry,
  OutputItem,
} from "@/components/contextual/OutputRenderers";
import type { OutputMetadata } from "@/components/contextual/OutputRenderers";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  ContentBadge,
  ContentCard,
  ContentCardHeader,
  ContentCardSubtitle,
  ContentCardTitle,
  ContentCodeBlock,
  ContentGrid,
  ContentLink,
  ContentMessage,
  ContentSuggestionsList,
} from "../../components/ToolAccordion/AccordionContent";
import {
  formatMaybeJson,
  getAnimationText,
  getViewAgentOutputToolOutput,
  isAgentOutputResponse,
  isErrorResponse,
  isNoResultsResponse,
  AccordionIcon,
  ToolIcon,
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

function isWorkspaceRef(value: unknown): value is string {
  return typeof value === "string" && value.startsWith("workspace://");
}

function resolveForRenderer(value: unknown): {
  value: unknown;
  metadata?: OutputMetadata;
} {
  if (!isWorkspaceRef(value)) return { value };

  const withoutPrefix = value.replace("workspace://", "");
  const fileId = withoutPrefix.split("#")[0];
  const apiPath = getGetWorkspaceDownloadFileByIdUrl(fileId);
  const url = `/api/proxy${apiPath}`;

  const hashIndex = value.indexOf("#");
  const mimeHint =
    hashIndex !== -1 ? value.slice(hashIndex + 1) || undefined : undefined;

  const metadata: OutputMetadata = {};
  if (mimeHint) {
    metadata.mimeType = mimeHint;
    if (mimeHint.startsWith("image/")) metadata.type = "image";
    else if (mimeHint.startsWith("video/")) metadata.type = "video";
  }

  return { value: url, metadata };
}

function RenderOutputValue({ value }: { value: unknown }) {
  const resolved = resolveForRenderer(value);
  const renderer = globalRegistry.getRenderer(
    resolved.value,
    resolved.metadata,
  );

  if (renderer) {
    return (
      <OutputItem
        value={resolved.value}
        metadata={resolved.metadata}
        renderer={renderer}
      />
    );
  }

  // Fallback for audio workspace refs
  if (
    isWorkspaceRef(value) &&
    resolved.metadata?.mimeType?.startsWith("audio/")
  ) {
    return (
      <audio controls src={String(resolved.value)} className="mt-2 w-full" />
    );
  }

  return null;
}

function getAccordionMeta(output: ViewAgentOutputToolOutput): {
  icon: React.ReactNode;
  title: string;
  description?: string;
} {
  const icon = <AccordionIcon />;

  if (isAgentOutputResponse(output)) {
    const status = output.execution?.status;
    return {
      icon,
      title: output.agent_name,
      description: status ? `Status: ${status}` : output.message,
    };
  }
  if (isNoResultsResponse(output)) {
    return { icon, title: "No results" };
  }
  return { icon, title: "Error" };
}

export function ViewAgentOutputTool({ part }: Props) {
  const text = getAnimationText(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  const output = getViewAgentOutputToolOutput(part);
  const isError =
    part.state === "output-error" || (!!output && isErrorResponse(output));
  const hasExpandableContent =
    part.state === "output-available" &&
    !!output &&
    (isAgentOutputResponse(output) ||
      isNoResultsResponse(output) ||
      isErrorResponse(output));

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
          {isAgentOutputResponse(output) && (
            <ContentGrid>
              <ContentCardHeader
                className="gap-3"
                action={
                  output.library_agent_link ? (
                    <ContentLink href={output.library_agent_link}>
                      Open
                    </ContentLink>
                  ) : null
                }
              >
                <ContentMessage>{output.message}</ContentMessage>
              </ContentCardHeader>

              {output.execution ? (
                <ContentGrid>
                  <ContentCard>
                    <ContentCardTitle className="text-xs">
                      Execution
                    </ContentCardTitle>
                    <ContentCardSubtitle className="mt-1">
                      {output.execution.execution_id}
                    </ContentCardSubtitle>
                    <ContentCardSubtitle className="mt-1">
                      Status: {output.execution.status}
                    </ContentCardSubtitle>
                  </ContentCard>

                  {output.execution.inputs_summary && (
                    <ContentCard>
                      <ContentCardTitle className="text-xs">
                        Inputs summary
                      </ContentCardTitle>
                      <div className="mt-2">
                        <RenderOutputValue
                          value={output.execution.inputs_summary}
                        />
                      </div>
                    </ContentCard>
                  )}

                  {Object.entries(output.execution.outputs ?? {}).map(
                    ([key, items]) => (
                      <ContentCard key={key}>
                        <div className="flex items-center justify-between gap-2">
                          <ContentCardTitle className="text-xs">
                            {key}
                          </ContentCardTitle>
                          <ContentBadge>
                            {items.length} item
                            {items.length === 1 ? "" : "s"}
                          </ContentBadge>
                        </div>
                        <div className="mt-2">
                          {items.slice(0, 3).map((item, i) => (
                            <RenderOutputValue key={i} value={item} />
                          ))}
                        </div>
                      </ContentCard>
                    ),
                  )}
                </ContentGrid>
              ) : (
                <ContentCard>
                  <ContentMessage>No execution selected.</ContentMessage>
                  <ContentCardSubtitle className="mt-1">
                    Try asking for a specific run or execution_id.
                  </ContentCardSubtitle>
                </ContentCard>
              )}
            </ContentGrid>
          )}

          {isNoResultsResponse(output) && (
            <ContentGrid>
              <ContentMessage>{output.message}</ContentMessage>
              {output.suggestions && output.suggestions.length > 0 && (
                <ContentSuggestionsList
                  items={output.suggestions}
                  className="mt-1"
                />
              )}
            </ContentGrid>
          )}

          {isErrorResponse(output) && (
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
