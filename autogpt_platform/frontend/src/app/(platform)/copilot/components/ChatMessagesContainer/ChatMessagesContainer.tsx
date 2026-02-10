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
import { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useState } from "react";
import { CreateAgentTool } from "../../tools/CreateAgent/CreateAgent";
import { EditAgentTool } from "../../tools/EditAgent/EditAgent";
import { FindAgentsTool } from "../../tools/FindAgents/FindAgents";
import { FindBlocksTool } from "../../tools/FindBlocks/FindBlocks";
import { RunAgentTool } from "../../tools/RunAgent/RunAgent";
import { RunBlockTool } from "../../tools/RunBlock/RunBlock";
import { SearchDocsTool } from "../../tools/SearchDocs/SearchDocs";
import { ViewAgentOutputTool } from "../../tools/ViewAgentOutput/ViewAgentOutput";

// ---------------------------------------------------------------------------
// Workspace media support
// ---------------------------------------------------------------------------

/**
 * Resolve workspace:// URLs in markdown text to proxy download URLs.
 * Detects MIME type from the hash fragment (e.g. workspace://id#video/mp4)
 * and prefixes the alt text with "video:" so the custom img component can
 * render a <video> element instead.
 */
function resolveWorkspaceUrls(text: string): string {
  return text.replace(
    /!\[([^\]]*)\]\(workspace:\/\/([^)#\s]+)(?:#([^)\s]*))?\)/g,
    (_match, alt: string, fileId: string, mimeHint?: string) => {
      const apiPath = getGetWorkspaceDownloadFileByIdUrl(fileId);
      const url = `/api/proxy${apiPath}`;
      if (mimeHint?.startsWith("video/")) {
        return `![video:${alt || "Video"}](${url})`;
      }
      return `![${alt || "Image"}](${url})`;
    },
  );
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
}

export const ChatMessagesContainer = ({
  messages,
  status,
  error,
  isLoading,
}: ChatMessagesContainerProps) => {
  const [thinkingPhrase, setThinkingPhrase] = useState(getRandomPhrase);

  useEffect(() => {
    if (status === "submitted") {
      setThinkingPhrase(getRandomPhrase());
    }
  }, [status]);

  const lastMessage = messages[messages.length - 1];
  const lastAssistantHasVisibleContent =
    lastMessage?.role === "assistant" &&
    lastMessage.parts.some(
      (p) =>
        (p.type === "text" && p.text.trim().length > 0) ||
        p.type.startsWith("tool-"),
    );

  const showThinking =
    status === "submitted" ||
    (status === "streaming" && !lastAssistantHasVisibleContent);

  return (
    <Conversation className="min-h-0 flex-1">
      <ConversationContent className="flex min-h-screen flex-1 flex-col gap-6 px-3 py-6">
        {isLoading && messages.length === 0 && (
          <div className="flex min-h-full flex-1 items-center justify-center">
            <LoadingSpinner className="text-neutral-600" />
          </div>
        )}
        {messages.map((message, messageIndex) => {
          const isLastAssistant =
            messageIndex === messages.length - 1 &&
            message.role === "assistant";
          const messageHasVisibleContent = message.parts.some(
            (p) =>
              (p.type === "text" && p.text.trim().length > 0) ||
              p.type.startsWith("tool-"),
          );

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
                    case "text":
                      return (
                        <MessageResponse
                          key={`${message.id}-${i}`}
                          components={STREAMDOWN_COMPONENTS}
                        >
                          {resolveWorkspaceUrls(part.text)}
                        </MessageResponse>
                      );
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
                    default:
                      return null;
                  }
                })}
                {isLastAssistant &&
                  !messageHasVisibleContent &&
                  showThinking && (
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
          <div className="rounded-lg bg-red-50 p-3 text-red-600">
            Error: {error.message}
          </div>
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
};
