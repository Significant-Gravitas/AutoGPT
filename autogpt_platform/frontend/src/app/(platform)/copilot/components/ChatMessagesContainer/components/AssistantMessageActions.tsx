"use client";

import {
  MessageAction,
  MessageActions,
} from "@/components/organisms/Message/Message";
import { cn } from "@/lib/utils";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { useMessageFeedback } from "../useMessageFeedback";
import { FeedbackModal } from "./FeedbackModal";
import { TTSButton } from "./TTSButton";
import {
  Copy02Icon,
  ThumbsDownIcon,
  ThumbsUpIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  message: UIMessage<unknown, UIDataTypes, UITools>;
  sessionID: string | null;
  savedMessageId?: string | null;
  className?: string;
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

export function AssistantMessageActions({
  message,
  sessionID,
  savedMessageId,
  className,
}: Props) {
  const {
    feedback,
    canRate,
    showFeedbackModal,
    handleCopy,
    handleUpvote,
    handleDownvoteClick,
    handleDownvoteSubmit,
    handleDownvoteCancel,
  } = useMessageFeedback({ sessionID, message, savedMessageId });

  const text = extractTextFromParts(message.parts);

  return (
    <>
      <MessageActions className={cn("mt-1", className)}>
        <MessageAction
          tooltip="Copy"
          onClick={() => handleCopy(text)}
          variant="ghost"
          size="icon-sm"
        >
          <Icon icon={Copy02Icon} size={16} />
        </MessageAction>

        <MessageAction
          tooltip="Good response"
          onClick={handleUpvote}
          variant="ghost"
          size="icon-sm"
          disabled={feedback === "downvote"}
          aria-disabled={!canRate || undefined}
          className={cn(
            feedback === "upvote" && "text-success/60 hover:text-success/60",
            feedback === "downvote" && "!opacity-20",
          )}
        >
          <span className={cn("flex", !canRate && "opacity-50")}>
            <Icon icon={ThumbsUpIcon} size={16} />
          </span>
        </MessageAction>

        <MessageAction
          tooltip="Bad response"
          onClick={handleDownvoteClick}
          variant="ghost"
          size="icon-sm"
          disabled={feedback === "upvote"}
          aria-disabled={!canRate || undefined}
          className={cn(
            feedback === "downvote" && "text-error/60 hover:text-error/60",
            feedback === "upvote" && "!opacity-20",
          )}
        >
          <span className={cn("flex", !canRate && "opacity-50")}>
            <Icon icon={ThumbsDownIcon} size={16} />
          </span>
        </MessageAction>

        <TTSButton text={text} sessionID={sessionID} />
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
