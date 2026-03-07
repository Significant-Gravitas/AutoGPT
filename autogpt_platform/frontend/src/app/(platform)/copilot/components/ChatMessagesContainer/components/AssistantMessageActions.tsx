"use client";

import {
  MessageAction,
  MessageActions,
} from "@/components/ai-elements/message";
import { cn } from "@/lib/utils";
import { CopySimple, ThumbsDown, ThumbsUp } from "@phosphor-icons/react";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { useMessageFeedback } from "../useMessageFeedback";
import { FeedbackModal } from "./FeedbackModal";
import { TTSButton } from "./TTSButton";

interface Props {
  message: UIMessage<unknown, UIDataTypes, UITools>;
  sessionID: string | null;
}

function extractTextFromParts(
  parts: UIMessage<unknown, UIDataTypes, UITools>["parts"],
): string {
  return parts
    .filter((p) => p.type === "text")
    .map((p) => (p as { type: "text"; text: string }).text)
    .join("\n")
    .trim();
}

export function AssistantMessageActions({ message, sessionID }: Props) {
  const {
    feedback,
    showFeedbackModal,
    handleCopy,
    handleUpvote,
    handleDownvoteClick,
    handleDownvoteSubmit,
    handleDownvoteCancel,
  } = useMessageFeedback({ sessionID, messageID: message.id });

  const text = extractTextFromParts(message.parts);

  return (
    <>
      <MessageActions className="mt-1">
        <MessageAction
          tooltip="Copy"
          onClick={() => handleCopy(text)}
          variant="ghost"
          size="icon-sm"
        >
          <CopySimple size={16} weight="regular" />
        </MessageAction>

        <MessageAction
          tooltip="Good response"
          onClick={handleUpvote}
          variant="ghost"
          size="icon-sm"
          disabled={feedback === "downvote"}
          className={cn(
            feedback === "upvote" && "text-green-300 hover:text-green-300",
            feedback === "downvote" && "!opacity-20",
          )}
        >
          <ThumbsUp
            size={16}
            weight={feedback === "upvote" ? "fill" : "regular"}
          />
        </MessageAction>

        <MessageAction
          tooltip="Bad response"
          onClick={handleDownvoteClick}
          variant="ghost"
          size="icon-sm"
          disabled={feedback === "upvote"}
          className={cn(
            feedback === "downvote" && "text-red-300 hover:text-red-300",
            feedback === "upvote" && "!opacity-20",
          )}
        >
          <ThumbsDown
            size={16}
            weight={feedback === "downvote" ? "fill" : "regular"}
          />
        </MessageAction>

        <TTSButton text={text} />
      </MessageActions>

      {showFeedbackModal && (
        <FeedbackModal
          isOpen={showFeedbackModal}
          onSubmit={handleDownvoteSubmit}
          onCancel={handleDownvoteCancel}
        />
      )}
    </>
  );
}
