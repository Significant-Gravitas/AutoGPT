import type { FileUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { cn } from "@/lib/utils";
import { parseSpecialMarkers } from "../../../../helpers/messageMarkers";
import { withToolDisplayNames } from "../../../../helpers/toolDisplay";
import type { TurnStats } from "../../../../helpers/convertChatSessionToUiMessages";
import {
  WORKSPACE_FOLDER_PART_TYPE,
  type WorkspaceFolderPartData,
} from "../../../../helpers/workspaceAttachments";
import { isBookkeepingPart } from "../../../../messageParts";
import type { SentFrom } from "../../../../sentFrom";
import {
  getLastCompactionCallId,
  getLatestCompactionPhase,
  getLatestCompactionStats,
  type MessagePart,
} from "../../helpers";
import { getVisibleUserMessageParts } from "../../userMessageParts";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;
type TextPart = Extract<MessagePart, { type: "text" }>;

export function deriveRowParts(
  role: ChatMessage["role"],
  parts: MessagePart[],
) {
  // Bookkeeping parts are stripped before any render/split logic so they
  // never reach the user UI, and so one landing between two tool calls
  // can't split a chain. data-status surfaces via ThinkingIndicator;
  // data-compaction via CompactionCard.
  const renderableParts = withToolDisplayNames(
    role === "user" ? getVisibleUserMessageParts(parts) : parts,
  ).filter((p) => !isBookkeepingPart(p));
  const textParts = renderableParts.filter(
    (p): p is TextPart => p.type === "text",
  );
  const lastTextPart = textParts[textParts.length - 1];
  const markerType =
    lastTextPart !== undefined
      ? parseSpecialMarkers(lastTextPart.text).markerType
      : null;
  const fileParts = renderableParts.filter(
    (p): p is FileUIPart => p.type === "file",
  );
  const folderParts = renderableParts.flatMap((p) =>
    p.type === WORKSPACE_FOLDER_PART_TYPE
      ? [(p as { data: WorkspaceFolderPartData }).data]
      : [],
  );
  return {
    renderableParts,
    textParts,
    copyText: textParts.map((p) => p.text).join("\n"),
    hasErrorMarker: markerType === "error" || markerType === "retryable_error",
    fileParts,
    folderParts,
  };
}

// Only a message that is actively streaming can have a live compaction
// phase — a stopped or failed turn must not leave an eternal progress bar.
// Replayed/settled messages never carry `data-compaction` parts, so they
// derive null either way.
export function deriveLiveCompaction(
  parts: MessagePart[],
  isCurrentlyStreaming: boolean,
) {
  const compactionPhase = isCurrentlyStreaming
    ? getLatestCompactionPhase(parts)
    : null;
  if (compactionPhase === null) {
    return {
      compactionPhase,
      liveCompactionCallId: null,
      liveCompactionStats: undefined,
    };
  }
  return {
    compactionPhase,
    // The phase belongs to the LAST compaction row only; earlier (settled)
    // rows in the same message stay settled.
    liveCompactionCallId: getLastCompactionCallId(parts),
    // Stats streamed on the `data-compaction` parts pace the live progress
    // curve while the tool row is still open (its own output stats only
    // exist once it closes).
    liveCompactionStats: getLatestCompactionStats(parts),
  };
}

export function getMessageContentClassName(compact: boolean): string {
  return cn(
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
  );
}

export function getRunPreview(parts: MessagePart[]): string {
  return parts
    .filter((p): p is TextPart => p.type === "text")
    .map((p) => p.text)
    .join(" ");
}

function isSameArray<T>(a: T[] | undefined, b: T[] | undefined): boolean {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  return a.every((item, i) => item === b[i]);
}

function isShallowEqual(
  a: object | null | undefined,
  b: object | null | undefined,
): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  const aKeys = Object.keys(a);
  if (aKeys.length !== Object.keys(b).length) return false;
  return aKeys.every((key) =>
    Object.is(a[key as keyof typeof a], b[key as keyof typeof b]),
  );
}

interface ComparableRowProps {
  turnMessages?: ChatMessage[];
  turnStat?: TurnStats;
  sessionSentFrom?: SentFrom | null;
}

// The parent rebuilds these three on every pass (a sliced turn, a stats
// entry from a rebuilt map, a session-level object), so they compare by
// content; every other prop compares by identity.
const VALUE_COMPARED_PROPS = new Set<string>([
  "turnMessages",
  "turnStat",
  "sessionSentFrom",
]);

export function areMessageRowPropsEqual<P extends ComparableRowProps>(
  prev: P,
  next: P,
): boolean {
  const keys = new Set([...Object.keys(prev), ...Object.keys(next)]);
  for (const key of keys) {
    if (VALUE_COMPARED_PROPS.has(key)) continue;
    if (!Object.is(prev[key as keyof P], next[key as keyof P])) return false;
  }
  return (
    isSameArray(prev.turnMessages, next.turnMessages) &&
    isShallowEqual(prev.turnStat, next.turnStat) &&
    isShallowEqual(prev.sessionSentFrom, next.sessionSentFrom)
  );
}
