import { memo, type ReactNode } from "react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import {
  Message,
  MessageActions,
  MessageContent,
} from "@/components/ai-elements/message";
import { cn } from "@/lib/utils";
import type { TurnStats } from "../../../../helpers/convertChatSessionToUiMessages";
import type { SentFrom } from "../../../../sentFrom";
import { TurnStatsBar } from "../../../JobStatsBar/TurnStatsBar";
import { WorkCard } from "../../../WorkCard/WorkCard";
import { AssistantMessageActions } from "../AssistantMessageActions";
import { ChainMessageParts } from "../ChainMessageParts";
import { CopyButton } from "../CopyButton";
import { MessageAttachments } from "../MessageAttachments";
import { MessagePartRenderer } from "../MessagePartRenderer";
import { QueueBadge } from "../QueueBadge";
import { SentFromBadge } from "../SentFromBadge";
import { UserMessageClamp } from "../UserMessageClamp";
import { areMessageRowPropsEqual } from "./helpers";
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

  const isAssistant = message.role === "assistant";
  const showActions =
    isLastInTurn &&
    !isCurrentlyStreaming &&
    textParts.length > 0 &&
    !hasErrorMarker;
  const createdAt = turnStat?.createdAt ? new Date(turnStat.createdAt) : null;

  return (
    <Message
      from={message.role}
      data-message-id={message.id}
      className="duration-300 animate-in fade-in slide-in-from-bottom-2 fill-mode-both"
    >
      <MessageContent
        className={cn(
          compact
            ? "text-sm leading-6 group-[.is-user]:rounded-xl"
            : "text-[1rem] leading-relaxed group-[.is-user]:rounded-3xl",
          "group-[.is-user]:bg-zinc-900 group-[.is-user]:px-4 group-[.is-user]:py-2.5 group-[.is-user]:text-zinc-50",
          // Streamdown's links (buttons, `text-primary`) and light
          // code chips would sink into the dark bubble.
          "group-[.is-user]:[&_[data-streamdown=link]]:break-all group-[.is-user]:[&_[data-streamdown=link]]:text-zinc-50 group-[.is-user]:[&_[data-streamdown=link]]:underline group-[.is-user]:[&_code]:bg-white/10 group-[.is-user]:[&_code]:text-zinc-50",
          "group-[.is-user]:[&_h1]:text-lg group-[.is-user]:[&_h1]:font-semibold group-[.is-user]:[&_h2]:text-lg group-[.is-user]:[&_h2]:font-semibold group-[.is-user]:[&_h3]:text-lg group-[.is-user]:[&_h3]:font-semibold group-[.is-user]:[&_h4]:text-lg group-[.is-user]:[&_h4]:font-semibold group-[.is-user]:[&_h5]:text-lg group-[.is-user]:[&_h5]:font-semibold group-[.is-user]:[&_h6]:text-lg group-[.is-user]:[&_h6]:font-semibold",
          // Chain hover pills use negative margins that the base
          // overflow-hidden would clip.
          "group-[.is-assistant]:overflow-visible group-[.is-assistant]:bg-transparent group-[.is-assistant]:text-slate-900",
        )}
      >
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
      {!readOnly &&
        message.role === "user" &&
        isSessionQueued &&
        turnStat?.isLatestUserMessage && (
          <MessageActions
            className="mt-1 items-center justify-end gap-1.5"
            data-testid="queue-status-row"
          >
            <QueueBadge sessionID={sessionID ?? null} />
          </MessageActions>
        )}
      {message.role === "user" && textParts.length > 0 && (
        <MessageActions className="mt-1 items-center justify-end gap-2 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
          {createdAt && !Number.isNaN(createdAt.getTime()) && (
            <span className="text-[11px] tabular-nums text-neutral-500">
              {createdAt.toLocaleString(undefined, {
                dateStyle: "medium",
                timeStyle: "short",
              })}
            </span>
          )}
          <CopyButton text={copyText} />
        </MessageActions>
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
      {!readOnly && showActions && (
        <AssistantMessageActions
          message={message}
          sessionID={sessionID ?? null}
          className={cn(
            !isLastAssistant &&
              "opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100",
          )}
        />
      )}
      {readOnly && showActions && (
        <MessageActions
          className={cn(
            "mt-1 items-center justify-start gap-2",
            !isLastAssistant &&
              "opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100",
          )}
        >
          <CopyButton text={copyText} />
        </MessageActions>
      )}
    </Message>
  );
}, areMessageRowPropsEqual);
