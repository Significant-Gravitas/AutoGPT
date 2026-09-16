import { isChainableToolPart, type MessagePart } from "../helpers";
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
  forceArtifacts?: boolean;
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
  forceArtifacts,
  readOnly,
  compactionPhase,
  liveCompactionCallId,
  liveCompactionStats,
}: Props) {
  const segments = buildChainSegments(parts, isChainableToolPart);
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
    return (
      <MessagePartRenderer
        key={`${messageID}-${segment.index}`}
        part={segment.part}
        messageID={messageID}
        partIndex={segment.index}
        onRetry={onRetry}
        fileUrlBuilder={fileUrlBuilder}
        forceArtifacts={forceArtifacts}
        readOnly={readOnly}
        compactionPhase={compactionPhase}
        liveCompactionCallId={liveCompactionCallId}
        liveCompactionStats={liveCompactionStats}
        isCurrentlyStreaming={isCurrentlyStreaming}
      />
    );
  });
}
