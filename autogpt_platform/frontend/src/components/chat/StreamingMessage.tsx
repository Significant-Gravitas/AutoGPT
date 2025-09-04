"use client";

import React, { useMemo } from "react";
import { cn } from "@/lib/utils";
import { User, Bot } from "lucide-react";
import { ToolCallWidget } from "./ToolCallWidget";
import { AgentCarousel } from "./AgentCarousel";
import { CredentialsSetupWidget } from "./CredentialsSetupWidget";
import { AgentSetupCard } from "./AgentSetupCard";

interface ContentSegment {
  type:
    | "text"
    | "tool"
    | "carousel"
    | "credentials_setup"
    | "agent_setup"
    | "auth_required";
  content: any;
  id?: string;
}

interface StreamingMessageProps {
  role: "USER" | "ASSISTANT" | "SYSTEM" | "TOOL";
  segments: ContentSegment[];
  className?: string;
  onSelectAgent?: (agent: any) => void;
  onGetAgentDetails?: (agent: any) => void;
}

export function StreamingMessage({
  role,
  segments,
  className,
  onSelectAgent,
  onGetAgentDetails,
}: StreamingMessageProps) {
  const isUser = role === "USER";
  const isAssistant = role === "ASSISTANT";
  const isSystem = role === "SYSTEM";
  const isTool = role === "TOOL";

  // Process segments to combine consecutive text segments
  const processedSegments = useMemo(() => {
    const result: ContentSegment[] = [];
    let currentText = "";

    segments.forEach((segment) => {
      if (segment.type === "text") {
        currentText += segment.content;
      } else {
        // Flush any accumulated text
        if (currentText) {
          result.push({ type: "text", content: currentText });
          currentText = "";
        }
        // Add the non-text segment
        result.push(segment);
      }
    });

    // Flush remaining text
    if (currentText) {
      result.push({ type: "text", content: currentText });
    }

    return result;
  }, [segments]);

  const renderSegment = (segment: ContentSegment, index: number) => {
    // Generate a unique key based on segment type, content hash, and index
    const segmentKey = `${segment.type}-${index}-${segment.id || segment.content?.id || Date.now()}`;

    switch (segment.type) {
      case "text":
        return (
          <div key={segmentKey} className="inline">
            {/* Simple markdown-like rendering */}
            {segment.content
              .split("\n")
              .map((line: string, lineIndex: number) => {
                const lineKey = `${segmentKey}-line-${lineIndex}`;
                if (line.startsWith("# ")) {
                  return (
                    <h1 key={lineKey} className="mb-2 text-xl font-bold">
                      {line.substring(2)}
                    </h1>
                  );
                } else if (line.startsWith("## ")) {
                  return (
                    <h2 key={lineKey} className="mb-2 text-lg font-bold">
                      {line.substring(3)}
                    </h2>
                  );
                } else if (line.startsWith("### ")) {
                  return (
                    <h3 key={lineKey} className="mb-2 text-base font-bold">
                      {line.substring(4)}
                    </h3>
                  );
                } else if (line.startsWith("- ")) {
                  return (
                    <li key={lineKey} className="ml-4 list-disc">
                      {line.substring(2)}
                    </li>
                  );
                } else if (line.startsWith("```")) {
                  return (
                    <pre
                      key={lineKey}
                      className="my-2 overflow-x-auto rounded bg-neutral-100 p-2 dark:bg-neutral-800"
                    >
                      <code>{line.substring(3)}</code>
                    </pre>
                  );
                } else if (line.trim() === "") {
                  return <br key={lineKey} />;
                } else {
                  return (
                    <span key={lineKey}>
                      {line}
                      {lineIndex < segment.content.split("\n").length - 1 &&
                        "\n"}
                    </span>
                  );
                }
              })}
          </div>
        );

      case "tool":
        const toolData = segment.content;
        return (
          <div key={segmentKey} className="my-3">
            <ToolCallWidget
              toolName={toolData.name}
              parameters={toolData.parameters}
              result={toolData.result}
              status={toolData.status}
              error={toolData.error}
            />
          </div>
        );

      case "carousel":
        const carouselData = segment.content;
        return (
          <div key={segmentKey} className="my-4">
            <AgentCarousel
              agents={carouselData.agents}
              query={carouselData.query}
              onSelectAgent={onSelectAgent!}
              onGetDetails={onGetAgentDetails!}
            />
          </div>
        );

      case "credentials_setup":
        const credentialsData = segment.content;
        return (
          <div key={segmentKey} className="my-4">
            <CredentialsSetupWidget
              agentId={credentialsData.agent_id}
              configuredCredentials={
                credentialsData.configured_credentials || []
              }
              missingCredentials={credentialsData.missing_credentials || []}
              totalRequired={credentialsData.total_required || 0}
              message={credentialsData.message}
            />
          </div>
        );

      case "agent_setup":
        const setupData = segment.content;
        return (
          <div key={segmentKey} className="my-4">
            <AgentSetupCard
              status={setupData.status}
              triggerType={setupData.trigger_type}
              name={setupData.name}
              graphId={setupData.graph_id}
              graphVersion={setupData.graph_version}
              scheduleId={setupData.schedule_id}
              webhookUrl={setupData.webhook_url}
              cron={setupData.cron}
              cronUtc={setupData.cron_utc}
              timezone={setupData.timezone}
              nextRun={setupData.next_run}
              addedToLibrary={setupData.added_to_library}
              libraryId={setupData.library_id}
              message={setupData.message}
            />
          </div>
        );

      case "auth_required":
        // Auth required segments are handled separately by ChatInterface
        // They trigger the auth prompt widget, not rendered inline
        return null;

      default:
        return null;
    }
  };

  return (
    <div
      className={cn(
        "flex gap-4 px-4 py-6",
        isUser && "justify-end",
        !isUser && "justify-start",
        className,
      )}
    >
      {!isUser && (
        <div className="flex-shrink-0">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-violet-600">
            <Bot className="h-5 w-5 text-white" />
          </div>
        </div>
      )}

      <div
        className={cn(
          "max-w-[70%]",
          isUser && "rounded-lg bg-neutral-100 px-4 py-3 dark:bg-neutral-800",
          isAssistant && "space-y-2",
          isSystem &&
            "rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 dark:border-blue-800 dark:bg-blue-900/20",
          isTool &&
            "rounded-lg border border-green-200 bg-green-50 px-4 py-3 dark:border-green-800 dark:bg-green-900/20",
        )}
      >
        {isSystem && (
          <div className="mb-2 text-xs font-medium text-blue-600 dark:text-blue-400">
            System
          </div>
        )}

        {isTool && (
          <div className="mb-2 text-xs font-medium text-green-600 dark:text-green-400">
            Tool Response
          </div>
        )}

        <div className="prose prose-sm dark:prose-invert max-w-none">
          {processedSegments.map(renderSegment)}
        </div>
      </div>

      {isUser && (
        <div className="flex-shrink-0">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-neutral-600">
            <User className="h-5 w-5 text-white" />
          </div>
        </div>
      )}
    </div>
  );
}
