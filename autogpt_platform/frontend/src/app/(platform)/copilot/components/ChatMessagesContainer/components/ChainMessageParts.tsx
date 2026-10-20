import { isChainableToolPart, type MessagePart } from "../helpers";
import type { ToolUIPart } from "ai";
import { useMemo } from "react";
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
  // Kept per `parts` so each ToolChain gets the same `parts` array back and
  // its own row derivation stays memoised.
  const segments = useMemo(
    () => buildChainSegments(parts, isChainableToolPart),
    [parts],
  );
  const lastChainSegmentIndex = segments.findLastIndex(
    (segment) => segment.kind === "chain",
  );

  return segments.map((segment, segmentIndex) => {
    if (segment.kind === "chain") {
      return (
        <ToolChain
          key={`${messageID}-chain-${segment.index}`}
          parts={segment.parts}
          isStreaming={
            isCurrentlyStreaming && segmentIndex === lastChainSegmentIndex
          }
          readOnly={readOnly}
        />
      );
    }
    if (segment.kind === "experts") {
      return (
        <ExpertChangeGroup
          key={`${messageID}-experts-${segment.index}`}
          parts={segment.parts as ToolUIPart[]}
          isCurrentlyStreaming={isCurrentlyStreaming}
          readOnly={readOnly}
        />
      );
    }
    return (
      <MessagePartRenderer
        key={`${messageID}-${segment.index}`}
        part={segment.part}
        messageID={messageID}
        partIndex={segment.index}
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
