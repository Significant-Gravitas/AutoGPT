import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useMemo } from "react";
import {
  getSentFromMetadata,
  isSessionOpeningMessage,
  type SentFrom,
} from "../../../../sentFrom";
import { getScheduledChatNotice } from "../../../ScheduledChatNoticeCard/helpers";
import { getWorkRunMetadata, toPreview } from "../../../WorkCard/helpers";
import { deriveLiveCompaction, deriveRowParts, getRunPreview } from "./helpers";

interface Args {
  message: UIMessage<unknown, UIDataTypes, UITools>;
  isCurrentlyStreaming: boolean;
  readOnly: boolean;
  sessionSentFrom: SentFrom | null;
}

export function useMessageRow({
  message,
  isCurrentlyStreaming,
  readOnly,
  sessionSentFrom,
}: Args) {
  const { role, parts, metadata } = message;
  // These memos stay with the compiler on. They are the only hooks here, so
  // without them this compiles as a plain function and caches nothing; and
  // the row would cache the whole result on every prop it gets, rebuilding
  // `renderableParts` when only, say, `isLastInTurn` changed.
  const runMetadata = useMemo(() => getWorkRunMetadata(metadata), [metadata]);
  const notice = useMemo(() => getScheduledChatNotice(metadata), [metadata]);
  const runPreview = useMemo(
    () => (runMetadata ? toPreview(getRunPreview(parts)) : ""),
    [runMetadata, parts],
  );
  const rowParts = useMemo(() => deriveRowParts(role, parts), [role, parts]);
  const compaction = useMemo(
    () => deriveLiveCompaction(parts, isCurrentlyStreaming),
    [parts, isCurrentlyStreaming],
  );

  const sentFrom = readOnly
    ? null
    : (getSentFromMetadata(metadata) ??
      (isSessionOpeningMessage(message) ? sessionSentFrom : null));

  return {
    runMetadata,
    runPreview,
    notice,
    ...rowParts,
    ...compaction,
    sentFrom,
  };
}
