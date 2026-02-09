"use client";

import type { ToolUIPart } from "ai";
import React, { useState } from "react";
import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
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

function resolveWorkspaceUrl(src: string): string {
  if (src.startsWith("workspace://")) {
    const withoutPrefix = src.replace("workspace://", "");
    const fileId = withoutPrefix.split("#")[0];
    const apiPath = getGetWorkspaceDownloadFileByIdUrl(fileId);
    return `/api/proxy${apiPath}`;
  }
  return src;
}

function getWorkspaceMimeHint(src: string): string | undefined {
  const hashIndex = src.indexOf("#");
  if (hashIndex === -1) return undefined;
  return src.slice(hashIndex + 1) || undefined;
}

function WorkspaceMedia({ value }: { value: string }) {
  const [imgFailed, setImgFailed] = useState(false);
  const resolvedUrl = resolveWorkspaceUrl(value);
  const mime = getWorkspaceMimeHint(value);

  if (mime?.startsWith("video/") || imgFailed) {
    return (
      <video
        controls
        className="mt-2 h-auto max-w-full rounded-md border border-zinc-200"
        preload="metadata"
      >
        <source src={resolvedUrl} />
      </video>
    );
  }

  if (mime?.startsWith("audio/")) {
    return <audio controls src={resolvedUrl} className="mt-2 w-full" />;
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={resolvedUrl}
      alt="Output media"
      className="mt-2 h-auto max-w-full rounded-md border border-zinc-200"
      loading="lazy"
      onError={() => setImgFailed(true)}
    />
  );
}

function isWorkspaceRef(value: unknown): value is string {
  return typeof value === "string" && value.startsWith("workspace://");
}

function renderOutputValue(value: unknown): React.ReactNode {
  if (isWorkspaceRef(value)) {
    return <WorkspaceMedia value={value} />;
  }
  if (Array.isArray(value)) {
    const workspaceItems = value.filter(isWorkspaceRef);
    if (workspaceItems.length > 0) {
      return (
        <>
          {value.map((item, i) =>
            isWorkspaceRef(item) ? (
              <WorkspaceMedia key={i} value={item} />
            ) : (
              <pre
                key={i}
                className="mt-1 whitespace-pre-wrap text-xs text-muted-foreground"
              >
                {formatMaybeJson(item)}
              </pre>
            ),
          )}
        </>
      );
    }
  }
  return null;
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
                      <pre className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">
                        {formatMaybeJson(output.execution.inputs_summary)}
                      </pre>
                    </ContentCard>
                  )}

                  {Object.entries(output.execution.outputs ?? {}).map(
                    ([key, items]) => {
                      const mediaContent = renderOutputValue(items);
                      return (
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
                          {mediaContent || (
                            <pre className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">
                              {formatMaybeJson(items.slice(0, 3))}
                            </pre>
                          )}
                        </ContentCard>
                      );
                    },
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
