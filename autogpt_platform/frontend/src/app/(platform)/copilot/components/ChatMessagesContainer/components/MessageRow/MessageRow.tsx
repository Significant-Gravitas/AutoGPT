import { memo, type ReactNode } from "react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { Message, MessageContent } from "@/components/ai-elements/message";
import type { TurnStats } from "../../../../helpers/convertChatSessionToUiMessages";
import type { SentFrom } from "../../../../sentFrom";
import { TurnStatsBar } from "../../../JobStatsBar/TurnStatsBar";
import { ScheduledChatNoticeCard } from "../../../ScheduledChatNoticeCard/ScheduledChatNoticeCard";
import { WorkCard } from "../../../WorkCard/WorkCard";
import { ChainMessageParts } from "../ChainMessageParts";
import { MessageAttachments } from "../MessageAttachments";
import { MessagePartRenderer } from "../MessagePartRenderer";
import { SentFromBadge } from "../SentFromBadge";
import { UserMessageClamp } from "../UserMessageClamp";
import { AssistantRowActions } from "./components/AssistantRowActions";
import { UserRowActions } from "./components/UserRowActions";
import { areMessageRowPropsEqual, getMessageContentClassName } from "./helpers";
import { useMessageRow } from "./useMessageRow";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

/**
 * One transcript row. Memoised: while a reply streams only the row whose
 * props change re-renders, so every prop must keep its identity between
 * passes unless the row really changed. Pass row-specific values (`undefined`
 * or `null` on rows they don't apply to) rather than session-wide ones.
 */
interface Props {
  /** The rendered row: a message, or one segment of a turn split at a
   *  mid-turn follow-up. */
  message: ChatMessage;
  /** Last rendered row and an assistant one. Only this row gets `onRetry`
   *  and keeps its actions visible without hover. */
  isLastAssistant: boolean;
  /** `isLastAssistant` while the chat status is streaming or submitted. */
  isCurrentlyStreaming: boolean;
  /** Assistant row that closes its turn: shows the stats bar and actions. */
  isLastInTurn: boolean;
  /** This row's entry from the turn stats map. */
  turnStat?: TurnStats;
  /** The turn this row closes, for the stats bar. Only set when the bar
   *  shows (`isLastInTurn` and not streaming). Compared by elements. */
  turnMessages?: ChatMessage[];
  /** Live elapsed seconds; only the last row gets them. */
  elapsedSeconds?: number;
  /** Thinking indicator rendered at the bottom of the last assistant row. */
  indicator?: ReactNode;
  /** User row while the session is queued; the queue badge shows on the
   *  latest user message. */
  isSessionQueued: boolean;
  /** Session-level provenance, used only by the row that opened the thread. */
  sessionSentFrom: SentFrom | null;
  sessionID?: string | null;
  readOnly: boolean;
  compact: boolean;
  filePattern?: RegExp;
  fileUrlBuilder?: (fileId: string) => string;
  onRetry?: () => void;
}

export const MessageRow = memo(function MessageRow({
  message,
  isLastAssistant,
  isCurrentlyStreaming,
  isLastInTurn,
  turnStat,
  turnMessages,
  elapsedSeconds,
  indicator,
  isSessionQueued,
  sessionSentFrom,
  sessionID,
  readOnly,
  compact,
  filePattern,
  fileUrlBuilder,
  onRetry,
}: Props) {
  const {
    runMetadata,
    runPreview,
    notice,
    renderableParts,
    textParts,
    copyText,
    hasErrorMarker,
    fileParts,
    folderParts,
    compactionPhase,
    liveCompactionCallId,
    liveCompactionStats,
    sentFrom,
  } = useMessageRow({
    message,
    isCurrentlyStreaming,
    readOnly,
    sessionSentFrom,
  });

  // A run-post rides structured metadata — render a compact WorkCard
  // instead of the raw markdown wall (legacy posts have no metadata and
  // fall through to normal rendering).
  if (runMetadata) {
    return (
      <Message
        from={message.role}
        data-message-id={message.id}
        className="duration-300 animate-in fade-in slide-in-from-bottom-2 fill-mode-both"
      >
        <MessageContent className="group-[.is-assistant]:bg-transparent">
          <WorkCard metadata={runMetadata} preview={runPreview} />
        </MessageContent>
      </Message>
    );
  }

  // A scheduled chat whose run was skipped says why, with the fix on a
  // button. The model never reads this row back, and it is the owner's: a
  // shared copy of the chat leaves it out.
  if (notice) {
    if (readOnly) return null;
    return (
      <Message
        from={message.role}
        data-message-id={message.id}
        className="duration-300 animate-in fade-in slide-in-from-bottom-2 fill-mode-both"
      >
        <MessageContent className="group-[.is-assistant]:bg-transparent">
          <ScheduledChatNoticeCard notice={notice} sessionID={sessionID} />
        </MessageContent>
      </Message>
    );
  }

  const isAssistant = message.role === "assistant";
  const showActions =
    isLastInTurn &&
    !isCurrentlyStreaming &&
    textParts.length > 0 &&
    !hasErrorMarker;

  return (
    <Message
      from={message.role}
      data-message-id={message.id}
      className="duration-300 animate-in fade-in slide-in-from-bottom-2 fill-mode-both"
    >
      <MessageContent className={getMessageContentClassName(compact)}>
        {isAssistant ? (
          <ChainMessageParts
            parts={renderableParts}
            messageID={message.id}
            isCurrentlyStreaming={isCurrentlyStreaming}
            onRetry={isLastAssistant ? onRetry : undefined}
            fileUrlBuilder={fileUrlBuilder}
            readOnly={readOnly}
            compactionPhase={compactionPhase}
            liveCompactionCallId={liveCompactionCallId}
            liveCompactionStats={liveCompactionStats}
          />
        ) : (
          <UserMessageClamp
            trailing={sentFrom ? <SentFromBadge sentFrom={sentFrom} /> : null}
          >
            {renderableParts.map((part, i) => (
              <MessagePartRenderer
                key={`${message.id}-${i}`}
                part={part}
                messageID={message.id}
                partIndex={i}
                fileUrlBuilder={fileUrlBuilder}
                readOnly={readOnly}
                compactionPhase={compactionPhase}
                liveCompactionCallId={liveCompactionCallId}
                liveCompactionStats={liveCompactionStats}
                isCurrentlyStreaming={isCurrentlyStreaming}
              />
            ))}
          </UserMessageClamp>
        )}
        {isLastInTurn && !isCurrentlyStreaming && (
          <TurnStatsBar
            turnMessages={turnMessages ?? [message]}
            elapsedSeconds={elapsedSeconds}
            stats={turnStat}
          />
        )}
        {indicator}
      </MessageContent>
      {message.role === "user" && (
        <UserRowActions
          showQueueBadge={
            !readOnly && isSessionQueued && !!turnStat?.isLatestUserMessage
          }
          showCopy={textParts.length > 0}
          createdAt={turnStat?.createdAt}
          copyText={copyText}
          sessionID={sessionID}
        />
      )}
      {(fileParts.length > 0 || folderParts.length > 0) && (
        <MessageAttachments
          files={fileParts}
          folders={folderParts}
          isUser={message.role === "user"}
          filePattern={filePattern}
          readOnly={readOnly}
        />
      )}
      {showActions && (
        <AssistantRowActions
          message={message}
          isLastAssistant={isLastAssistant}
          readOnly={readOnly}
          copyText={copyText}
          sessionID={sessionID}
        />
      )}
    </Message>
  );
}, areMessageRowPropsEqual);
