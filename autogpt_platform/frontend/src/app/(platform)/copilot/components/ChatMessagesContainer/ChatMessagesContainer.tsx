import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useState } from "react";
import { CreateAgentTool } from "../../tools/CreateAgent/CreateAgent";
import { EditAgentTool } from "../../tools/EditAgent/EditAgent";
import {
  CreateFeatureRequestTool,
  SearchFeatureRequestsTool,
} from "../../tools/FeatureRequests/FeatureRequests";
import { FindAgentsTool } from "../../tools/FindAgents/FindAgents";
import { FindBlocksTool } from "../../tools/FindBlocks/FindBlocks";
import { RunAgentTool } from "../../tools/RunAgent/RunAgent";
import { RunBlockTool } from "../../tools/RunBlock/RunBlock";
import { SearchDocsTool } from "../../tools/SearchDocs/SearchDocs";
import { GenericTool } from "../../tools/GenericTool/GenericTool";
import { ViewAgentOutputTool } from "../../tools/ViewAgentOutput/ViewAgentOutput";

// ---------------------------------------------------------------------------
// Special text parsing (error markers, workspace URLs, etc.)
// ---------------------------------------------------------------------------

// Special message prefixes for text-based markers (set by backend)
const COPILOT_ERROR_PREFIX = "[COPILOT_ERROR]";
const COPILOT_SYSTEM_PREFIX = "[COPILOT_SYSTEM]";

type MarkerType = "error" | "system" | null;

/**
 * Parse special markers from message content (error, system).
 *
 * Detects markers added by the backend for special rendering:
 * - `[COPILOT_ERROR] message` → ErrorCard
 * - `[COPILOT_SYSTEM] message` → System info message
 *
 * Returns marker type, marker text, and cleaned text.
 */
function parseSpecialMarkers(text: string): {
  markerType: MarkerType;
  markerText: string;
  cleanText: string;
} {
  // Check for error marker
  const errorMatch = text.match(
    new RegExp(`\\${COPILOT_ERROR_PREFIX}\\s*(.+?)$`, "s"),
  );
  if (errorMatch) {
    return {
      markerType: "error",
      markerText: errorMatch[1].trim(),
      cleanText: text.replace(errorMatch[0], "").trim(),
    };
  }

  // Check for system marker
  const systemMatch = text.match(
    new RegExp(`\\${COPILOT_SYSTEM_PREFIX}\\s*(.+?)$`, "s"),
  );
  if (systemMatch) {
    return {
      markerType: "system",
      markerText: systemMatch[1].trim(),
      cleanText: text.replace(systemMatch[0], "").trim(),
    };
  }

  return { markerType: null, markerText: "", cleanText: text };
}

/**
 * Resolve workspace:// URLs in markdown text to proxy download URLs.
 *
 * Handles both image syntax  `![alt](workspace://id#mime)` and regular link
 * syntax `[text](workspace://id)`.  For images the MIME type hash fragment is
 * inspected so that videos can be rendered with a `<video>` element via the
 * custom img component.
 */
function resolveWorkspaceUrls(text: string): string {
  // Handle image links: ![alt](workspace://id#mime)
  let resolved = text.replace(
    /!\[([^\]]*)\]\(workspace:\/\/([^)#\s]+)(?:#([^)#\s]*))?\)/g,
    (_match, alt: string, fileId: string, mimeHint?: string) => {
      const apiPath = getGetWorkspaceDownloadFileByIdUrl(fileId);
      const url = `/api/proxy${apiPath}`;
      if (mimeHint?.startsWith("video/")) {
        return `![video:${alt || "Video"}](${url})`;
      }
      return `![${alt || "Image"}](${url})`;
    },
  );

  // Handle regular links: [text](workspace://id) — without the leading "!"
  // These are blocked by Streamdown's rehype-harden sanitizer because
  // "workspace://" is not in the allowed URL-scheme whitelist, which causes
  // "[blocked]" to appear next to the link text.
  // Use an absolute URL so Streamdown's "Copy link" button copies the full
  // URL (including host) rather than just the path.
  resolved = resolved.replace(
    /(?<!!)\[([^\]]*)\]\(workspace:\/\/([^)#\s]+)(?:#[^)#\s]*)?\)/g,
    (_match, linkText: string, fileId: string) => {
      const apiPath = getGetWorkspaceDownloadFileByIdUrl(fileId);
      const origin =
        typeof window !== "undefined" ? window.location.origin : "";
      const url = `${origin}/api/proxy${apiPath}`;
      return `[${linkText || "Download file"}](${url})`;
    },
  );

  return resolved;
}

/**
 * Custom img component for Streamdown that renders <video> elements
 * for workspace video files (detected via "video:" alt-text prefix).
 * Falls back to <video> when an <img> fails to load for workspace files.
 */
function WorkspaceMediaImage(props: React.JSX.IntrinsicElements["img"]) {
  const { src, alt, ...rest } = props;
  const [imgFailed, setImgFailed] = useState(false);
  const isWorkspace = src?.includes("/workspace/files/") ?? false;

  if (!src) return null;

  if (alt?.startsWith("video:") || (imgFailed && isWorkspace)) {
    return (
      <span className="my-2 inline-block">
        <video
          controls
          className="h-auto max-w-full rounded-md border border-zinc-200"
          preload="metadata"
        >
          <source src={src} />
          Your browser does not support the video tag.
        </video>
      </span>
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={alt || "Image"}
      className="h-auto max-w-full rounded-md border border-zinc-200"
      loading="lazy"
      onError={() => {
        if (isWorkspace) setImgFailed(true);
      }}
      {...rest}
    />
  );
}

/** Stable components override for Streamdown (avoids re-creating on every render). */
const STREAMDOWN_COMPONENTS = { img: WorkspaceMediaImage };

const THINKING_PHRASES = [
  "Thinking...",
  "Considering this...",
  "Working through this...",
  "Analyzing your request...",
  "Reasoning...",
  "Looking into it...",
  "Processing your request...",
  "Mulling this over...",
  "Piecing it together...",
  "On it...",
];

function getRandomPhrase() {
  return THINKING_PHRASES[Math.floor(Math.random() * THINKING_PHRASES.length)];
}

interface ChatMessagesContainerProps {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  status: string;
  error: Error | undefined;
  isLoading: boolean;
  headerSlot?: React.ReactNode;
}

export const ChatMessagesContainer = ({
  messages,
  status,
  error,
  isLoading,
  headerSlot,
}: ChatMessagesContainerProps) => {
  const [thinkingPhrase, setThinkingPhrase] = useState(getRandomPhrase);

  const lastMessage = messages[messages.length - 1];

  // Determine if something is visibly "in-flight" in the last assistant message:
  // - Text is actively streaming (last part is non-empty text)
  // - A tool call is pending (state is input-streaming or input-available)
  const hasInflight = (() => {
    if (lastMessage?.role !== "assistant") return false;
    const parts = lastMessage.parts;
    if (parts.length === 0) return false;

    const lastPart = parts[parts.length - 1];

    // Text is actively being written
    if (lastPart.type === "text" && lastPart.text.trim().length > 0)
      return true;

    // A tool call is still pending (no output yet)
    if (
      lastPart.type.startsWith("tool-") &&
      "state" in lastPart &&
      (lastPart.state === "input-streaming" ||
        lastPart.state === "input-available")
    )
      return true;

    return false;
  })();

  const showThinking =
    status === "submitted" || (status === "streaming" && !hasInflight);

  useEffect(() => {
    if (showThinking) {
      setThinkingPhrase(getRandomPhrase());
    }
  }, [showThinking]);

  return (
    <Conversation className="min-h-0 flex-1">
      <ConversationContent className="flex flex-1 flex-col gap-6 px-3 py-6">
        {headerSlot}
        {isLoading && messages.length === 0 && (
          <div
            className="flex flex-1 items-center justify-center"
            style={{ minHeight: "calc(100vh - 12rem)" }}
          >
            <LoadingSpinner className="text-neutral-600" />
          </div>
        )}
        {messages.map((message, messageIndex) => {
          const isLastAssistant =
            messageIndex === messages.length - 1 &&
            message.role === "assistant";

          return (
            <Message from={message.role} key={message.id}>
              <MessageContent
                className={
                  "text-[1rem] leading-relaxed " +
                  "group-[.is-user]:rounded-xl group-[.is-user]:bg-purple-100 group-[.is-user]:px-3 group-[.is-user]:py-2.5 group-[.is-user]:text-slate-900 group-[.is-user]:[border-bottom-right-radius:0] " +
                  "group-[.is-assistant]:bg-transparent group-[.is-assistant]:text-slate-900"
                }
              >
                {message.parts.map((part, i) => {
                  switch (part.type) {
                    case "text": {
                      // Check for special markers (error, system)
                      const { markerType, markerText, cleanText } =
                        parseSpecialMarkers(part.text);

                      if (markerType === "error") {
                        return (
                          <ErrorCard
                            key={`${message.id}-${i}`}
                            responseError={{ message: markerText }}
                            context="execution"
                          />
                        );
                      }

                      if (markerType === "system") {
                        return (
                          <div
                            key={`${message.id}-${i}`}
                            className="my-2 rounded-lg bg-neutral-100 px-3 py-2 text-sm italic text-neutral-600"
                          >
                            {markerText}
                          </div>
                        );
                      }

                      return (
                        <MessageResponse
                          key={`${message.id}-${i}`}
                          components={STREAMDOWN_COMPONENTS}
                        >
                          {resolveWorkspaceUrls(cleanText)}
                        </MessageResponse>
                      );
                    }
                    case "tool-find_block":
                      return (
                        <FindBlocksTool
                          key={`${message.id}-${i}`}
                          part={part as ToolUIPart}
                        />
                      );
                    case "tool-find_agent":
                    case "tool-find_library_agent":
                      return (
                        <FindAgentsTool
                          key={`${message.id}-${i}`}
                          part={part as ToolUIPart}
                        />
                      );
                    case "tool-search_docs":
                    case "tool-get_doc_page":
                      return (
                        <SearchDocsTool
                          key={`${message.id}-${i}`}
                          part={part as ToolUIPart}
                        />
                      );
                    case "tool-run_block":
                      return (
                        <RunBlockTool
                          key={`${message.id}-${i}`}
                          part={part as ToolUIPart}
                        />
                      );
                    case "tool-run_agent":
                    case "tool-schedule_agent":
                      return (
                        <RunAgentTool
                          key={`${message.id}-${i}`}
                          part={part as ToolUIPart}
                        />
                      );
                    case "tool-create_agent":
                      return (
                        <CreateAgentTool
                          key={`${message.id}-${i}`}
                          part={part as ToolUIPart}
                        />
                      );
                    case "tool-edit_agent":
                      return (
                        <EditAgentTool
                          key={`${message.id}-${i}`}
                          part={part as ToolUIPart}
                        />
                      );
                    case "tool-view_agent_output":
                      return (
                        <ViewAgentOutputTool
                          key={`${message.id}-${i}`}
                          part={part as ToolUIPart}
                        />
                      );
                    case "tool-search_feature_requests":
                      return (
                        <SearchFeatureRequestsTool
                          key={`${message.id}-${i}`}
                          part={part as ToolUIPart}
                        />
                      );
                    case "tool-create_feature_request":
                      return (
                        <CreateFeatureRequestTool
                          key={`${message.id}-${i}`}
                          part={part as ToolUIPart}
                        />
                      );
                    default:
                      // Render a generic tool indicator for SDK built-in
                      // tools (Read, Glob, Grep, etc.) or any unrecognized tool
                      if (part.type.startsWith("tool-")) {
                        return (
                          <GenericTool
                            key={`${message.id}-${i}`}
                            part={part as ToolUIPart}
                          />
                        );
                      }
                      return null;
                  }
                })}
                {isLastAssistant && showThinking && (
                  <span className="inline-block animate-shimmer bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-[length:200%_100%] bg-clip-text text-transparent">
                    {thinkingPhrase}
                  </span>
                )}
              </MessageContent>
            </Message>
          );
        })}
        {showThinking && lastMessage?.role !== "assistant" && (
          <Message from="assistant">
            <MessageContent className="text-[1rem] leading-relaxed">
              <span className="inline-block animate-shimmer bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-[length:200%_100%] bg-clip-text text-transparent">
                {thinkingPhrase}
              </span>
            </MessageContent>
          </Message>
        )}
        {error && (
          <details className="rounded-lg bg-red-50 p-4 text-sm text-red-700">
            <summary className="cursor-pointer font-medium">
              The assistant encountered an error. Please try sending your
              message again.
            </summary>
            <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words text-xs text-red-600">
              {error instanceof Error ? error.message : String(error)}
            </pre>
          </details>
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
};
