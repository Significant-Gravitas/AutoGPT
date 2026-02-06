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
import { UIDataTypes, UIMessage, UITools, ToolUIPart } from "ai";
import { FindBlocksTool } from "../../tools/FindBlocks/FindBlocks";
import { FindAgentsTool } from "../../tools/FindAgents/FindAgents";
import { SearchDocsTool } from "../../tools/SearchDocs/SearchDocs";
import { RunBlockTool } from "../../tools/RunBlock/RunBlock";
import { RunAgentTool } from "../../tools/RunAgent/RunAgent";
import { ViewAgentOutputTool } from "../../tools/ViewAgentOutput/ViewAgentOutput";
import { CreateAgentTool } from "../../tools/CreateAgent/CreateAgent";
import { EditAgentTool } from "../../tools/EditAgent/EditAgent";

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
  return (
    <Conversation className="min-h-0 flex-1">
      <ConversationContent className="gap-6 px-3 py-6">
        {isLoading && messages.length === 0 && (
          <div className="flex flex-1 items-center justify-center">
            <LoadingSpinner size="large" className="text-neutral-400" />
          </div>
        )}
        {messages.map((message) => (
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
                      <MessageResponse key={`${message.id}-${i}`}>
                        {part.text}
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
            </MessageContent>
          </Message>
        ))}
        {status === "submitted" && (
          <Message from="assistant">
            <MessageContent className="text-[1rem] leading-relaxed">
              <span className="inline-block animate-shimmer bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-[length:200%_100%] bg-clip-text text-transparent">
                Thinking...
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
