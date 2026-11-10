import { isChainableToolPart, type MessagePart } from "../helpers";
import type { ToolUIPart } from "ai";
import { useContext, useMemo } from "react";
import { HeldOutcomesContext } from "../HeldOutcomesContext";
import { withHeldOutcome } from "../heldCallRows";
import { ExpertChangeGroup } from "../../ToolChain/ExpertCards";
import { buildChainSegments } from "../../ToolChain/helpers";
import { ToolChain } from "../../ToolChain/ToolChain";
import type {
  CompactionPhase,
  CompactionStats,
} from "../../CompactionCard/helpers";
import { MessagePartRenderer } from "./MessagePartRenderer";

interface Props {
  parts: MessagePart[];
  messageID: string;
  isCurrentlyStreaming: boolean;
  onRetry?: () => void;
  fileUrlBuilder?: (fileId: string) => string;
  readOnly?: boolean;
  compactionPhase?: CompactionPhase | null;
  liveCompactionCallId?: string | null;
  liveCompactionStats?: CompactionStats;
}

export function ChainMessageParts({
  parts,
  messageID,
  isCurrentlyStreaming,
  onRetry,
  fileUrlBuilder,
  readOnly,
  compactionPhase,
  liveCompactionCallId,
  liveCompactionStats,
}: Props) {
  const outcomes = useContext(HeldOutcomesContext);
  // Kept per `parts` so each ToolChain gets the same `parts` array back and
  // its own row derivation stays memoised. A manual memo even with the
  // compiler on, which would rebuild this with the rows below whenever any
  // prop or the outcomes context changes.
  const segments = useMemo(
    () => buildChainSegments(parts, isChainableToolPart),
    [parts],
  );
  const lastChainSegmentIndex = segments.findLastIndex(
    (segment) => segment.kind === "chain",
  );
  const ordinals = segmentOrdinals(segments);

  return segments.map((segment, segmentIndex) => {
    const key = `${messageID}-${segmentKey(segment, ordinals[segmentIndex])}`;
    if (segment.kind === "chain") {
      return (
        <ToolChain
          key={key}
          parts={segment.parts}
          isStreaming={
            isCurrentlyStreaming && segmentIndex === lastChainSegmentIndex
          }
          readOnly={readOnly}
        />
      );
    }
    // Outside the chain a part renders from its held call's outcome; chain rows
    // keep the stub, which applyHeldOutcome words.
    if (segment.kind === "experts") {
      return (
        <ExpertChangeGroup
          key={key}
          parts={
            segment.parts.map((part) =>
              withHeldOutcome(part, outcomes),
            ) as ToolUIPart[]
          }
          isCurrentlyStreaming={isCurrentlyStreaming}
          readOnly={readOnly}
        />
      );
    }
    return (
      <MessagePartRenderer
        key={key}
        part={withHeldOutcome(segment.part, outcomes)}
        messageID={messageID}
        partIndex={ordinals[segmentIndex]}
        onRetry={onRetry}
        fileUrlBuilder={fileUrlBuilder}
        readOnly={readOnly}
        compactionPhase={compactionPhase}
        liveCompactionCallId={liveCompactionCallId}
        liveCompactionStats={liveCompactionStats}
        isCurrentlyStreaming={isCurrentlyStreaming}
      />
    );
  });
}

type ChainSegment = ReturnType<typeof buildChainSegments>[number];

// Keyed by what a segment holds, never by its part index: the parts a live
// message carries (step markers, bookkeeping, blank text) are not in the
// stored rows a reload or the end-of-stream hydration builds, so the indexes
// move while the segments stay the same, and a moved key remounts the chain.
function segmentKey(segment: ChainSegment, ordinal: number) {
  const first = segment.kind === "part" ? segment.part : segment.parts[0];
  const callID =
    first && "toolCallId" in first && typeof first.toolCallId === "string"
      ? first.toolCallId
      : null;
  return callID ? `${segment.kind}-${callID}` : `${segment.kind}-${ordinal}`;
}

/** Each segment's position among the segments of its kind. */
function segmentOrdinals(segments: ChainSegment[]) {
  const seen = new Map<string, number>();
  return segments.map((segment) => {
    const ordinal = seen.get(segment.kind) ?? 0;
    seen.set(segment.kind, ordinal + 1);
    return ordinal;
  });
}
