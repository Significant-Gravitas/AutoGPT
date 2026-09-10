import { MessageResponse } from "@/components/ai-elements/message";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { StoppedTaskCard } from "./StoppedTaskCard";
import { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { ArtifactCard } from "../../ArtifactCard/ArtifactCard";
import { AskQuestionTool } from "../../../tools/AskQuestion/AskQuestion";
import { ConnectIntegrationTool } from "../../../tools/ConnectIntegrationTool/ConnectIntegrationTool";
import { CreateAgentTool } from "../../../tools/CreateAgent/CreateAgent";
import { DecomposeGoalTool } from "../../../tools/DecomposeGoal/DecomposeGoal";
import { EditAgentTool } from "../../../tools/EditAgent/EditAgent";
import {
  CreateFeatureRequestTool,
  SearchFeatureRequestsTool,
} from "../../../tools/FeatureRequests/FeatureRequests";
import { FindAgentsTool } from "../../../tools/FindAgents/FindAgents";
import { FolderTool } from "../../../tools/FolderTool/FolderTool";
import { FindBlocksTool } from "../../../tools/FindBlocks/FindBlocks";
import { GenericTool } from "../../../tools/GenericTool/GenericTool";
import { RunAgentTool } from "../../../tools/RunAgent/RunAgent";
import { RunBlockTool } from "../../../tools/RunBlock/RunBlock";
import { RunMCPToolComponent } from "../../../tools/RunMCPTool/RunMCPTool";
import { SearchDocsTool } from "../../../tools/SearchDocs/SearchDocs";
import { SetupTriggerTool } from "../../../tools/SetupTrigger/SetupTrigger";
import { ViewAgentOutputTool } from "../../../tools/ViewAgentOutput/ViewAgentOutput";
import { CompactionCard } from "../../CompactionCard/CompactionCard";
import {
  parseCompactionOutput,
  type CompactionPhase,
  type CompactionStats,
} from "../../CompactionCard/helpers";
import { COMPACTION_PART_TYPE } from "../../ToolChain/helpers";
import {
  extractWorkspaceArtifacts,
  isRetiredCompactionRow,
  parseSpecialMarkers,
  resolveWorkspaceUrls,
} from "../helpers";
import { ReasoningCollapse } from "./ReasoningCollapse";

/**
 * Custom img component for Streamdown that renders <video> elements
 * for workspace video files (detected via "video:" alt-text prefix).
 * Falls back to <video> when an <img> fails to load for workspace files.
 */
function WorkspaceMediaImage(props: React.JSX.IntrinsicElements["img"]) {
  const { src, alt, ...rest } = props;

  if (!src) return null;

  if (alt?.startsWith("video:")) {
    return (
      <span className="my-2 inline-block">
        <video
          controls
          className="h-auto max-w-full rounded-md border border-zinc-200"
          preload="metadata"
        >
          <source src={src} />
          Your browser does not support the video tag.
        </video>
      </span>
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={alt || "Image"}
      className="h-auto max-w-full rounded-md border border-zinc-200"
      loading="lazy"
      {...rest}
    />
  );
}

/** Stable components override for Streamdown (avoids re-creating on every render). */
const STREAMDOWN_COMPONENTS = { img: WorkspaceMediaImage };

function TextWithArtifactCards({
  text,
  fileUrlBuilder,
  forceArtifacts,
  readOnly,
}: {
  text: string;
  fileUrlBuilder?: (fileId: string) => string;
  forceArtifacts?: boolean;
  readOnly?: boolean;
}) {
  const isArtifactsFlagEnabled = useGetFlag(Flag.ARTIFACTS);
  const isArtifactsEnabled = forceArtifacts || isArtifactsFlagEnabled;
  const artifacts = extractWorkspaceArtifacts(text, fileUrlBuilder);
  const resolved = resolveWorkspaceUrls(text, fileUrlBuilder);

  // Text reads first, with the artifact cards trailing.
  return (
    <>
      <MessageResponse
        components={STREAMDOWN_COMPONENTS}
        className="[&_li]:py-0"
      >
        {resolved}
      </MessageResponse>
      {isArtifactsEnabled && artifacts.length > 0 && (
        <div className="mt-2 grid grid-cols-1 gap-1 sm:grid-cols-2">
          {artifacts.map((artifact) => (
            <ArtifactCard
              key={artifact.id}
              artifact={artifact}
              readOnly={readOnly}
            />
          ))}
        </div>
      )}
    </>
  );
}

interface Props {
  part: UIMessage<unknown, UIDataTypes, UITools>["parts"][number];
  messageID: string;
  partIndex: number;
  onRetry?: () => void;
  /** Override the URL emitted when rewriting workspace:// references
   *  in markdown.  Owner side defaults to the workspace-file endpoint;
   *  the public share viewer passes a token-aware builder so anonymous
   *  readers can download via the public allowlist-gated route. */
  fileUrlBuilder?: (fileId: string) => string;
  /** Force inline artifact-card rendering for workspace:// URIs in
   *  prose, regardless of the ``ARTIFACTS`` LD flag. */
  forceArtifacts?: boolean;
  /** Read-only mode — forwarded so embedded ``ArtifactCard``s
   *  download on click instead of opening a panel. */
  readOnly?: boolean;
  /** Live `data-compaction` phase for the enclosing message, derived by
   *  the caller from the message's parts. Drives the compaction row's
   *  progress bar; null once the row has settled into history. */
  compactionPhase?: CompactionPhase | null;
  /** Tool-call ID of the message's last compaction row — the only row the
   *  live phase applies to. Earlier (settled) rows render as history even
   *  while a later cycle streams its phases. */
  liveCompactionCallId?: string | null;
  /** Stats streamed on the message's `data-compaction` parts. They pace the
   *  live progress curve before the tool row closes; once it does, the
   *  row's own parsed output wins. */
  liveCompactionStats?: CompactionStats;
  /** Whether the enclosing message is still streaming. A compaction row
   *  only animates while it is; once the stream ends the row is history,
   *  however it was left. */
  isCurrentlyStreaming?: boolean;
}

export function MessagePartRenderer({
  part,
  messageID,
  partIndex,
  onRetry,
  fileUrlBuilder,
  forceArtifacts,
  readOnly,
  compactionPhase,
  liveCompactionCallId,
  liveCompactionStats,
  isCurrentlyStreaming,
}: Props) {
  const key = `${messageID}-${partIndex}`;

  switch (part.type) {
    case "reasoning": {
      const reasoningText =
        "text" in part && typeof part.text === "string" ? part.text : "";
      if (!reasoningText.trim()) return null;
      // AI SDK reasoning parts carry an optional `state: "streaming" | "done"`.
      // We pulse the indicator only while streaming so a finalized reasoning
      // block doesn't keep looking like the model is still thinking.
      const reasoningState =
        "state" in part && typeof part.state === "string" ? part.state : null;
      const isActive = reasoningState === "streaming";
      return (
        <ReasoningCollapse key={key} isActive={isActive}>
          <pre className="whitespace-pre-wrap text-sm text-zinc-700">
            {reasoningText}
          </pre>
        </ReasoningCollapse>
      );
    }
    case "text": {
      const { markerType, markerText, cleanText } = parseSpecialMarkers(
        part.text,
      );

      if (markerType === "error" || markerType === "retryable_error") {
        const lowerMarker = markerText.toLowerCase();
        const isCancellation =
          lowerMarker === "operation cancelled" ||
          lowerMarker === "execution stopped by user";
        if (isCancellation) {
          return <StoppedTaskCard key={key} />;
        }
        return (
          <ErrorCard
            key={key}
            responseError={{ message: markerText }}
            context="execution"
            onRetry={markerType === "retryable_error" ? onRetry : undefined}
          />
        );
      }

      if (markerType === "system") {
        return (
          <div
            key={key}
            className="my-2 rounded-lg bg-neutral-100 px-3 py-2 text-sm italic text-neutral-600"
          >
            {markerText}
          </div>
        );
      }

      return (
        <TextWithArtifactCards
          key={key}
          text={cleanText}
          fileUrlBuilder={fileUrlBuilder}
          forceArtifacts={forceArtifacts}
          readOnly={readOnly}
        />
      );
    }
    case "tool-ask_question":
      return <AskQuestionTool key={key} part={part as ToolUIPart} />;
    case "tool-find_block":
      return <FindBlocksTool key={key} part={part as ToolUIPart} />;
    case "tool-find_agent":
    case "tool-find_library_agent":
      return <FindAgentsTool key={key} part={part as ToolUIPart} />;
    case "tool-search_docs":
    case "tool-get_doc_page":
      return <SearchDocsTool key={key} part={part as ToolUIPart} />;
    case "tool-connect_integration":
      return <ConnectIntegrationTool key={key} part={part as ToolUIPart} />;
    case "tool-run_block":
    case "tool-continue_run_block":
      return <RunBlockTool key={key} part={part as ToolUIPart} />;
    case "tool-run_mcp_tool":
      return <RunMCPToolComponent key={key} part={part as ToolUIPart} />;
    case "tool-run_agent":
    case "tool-schedule_agent":
      return <RunAgentTool key={key} part={part as ToolUIPart} />;
    case "tool-setup_agent_webhook_trigger":
      return <SetupTriggerTool key={key} part={part as ToolUIPart} />;
    case "tool-decompose_goal":
      return <DecomposeGoalTool key={key} part={part as ToolUIPart} />;
    case "tool-create_agent":
      return <CreateAgentTool key={key} part={part as ToolUIPart} />;
    case "tool-edit_agent":
      return <EditAgentTool key={key} part={part as ToolUIPart} />;
    case "tool-view_agent_output":
      return <ViewAgentOutputTool key={key} part={part as ToolUIPart} />;
    case "tool-search_feature_requests":
      return <SearchFeatureRequestsTool key={key} part={part as ToolUIPart} />;
    case "tool-create_feature_request":
      return <CreateFeatureRequestTool key={key} part={part as ToolUIPart} />;
    case "tool-create_folder":
    case "tool-list_folders":
    case "tool-update_folder":
    case "tool-move_folder":
    case "tool-delete_folder":
    case "tool-move_agents_to_folder":
      return <FolderTool key={key} part={part as ToolUIPart} />;
    case "tool-TodoWrite":
      // Hidden inline — the task list surfaces through TaskProgressBar above
      // the composer, not as a message part. That bar is gated on
      // TASK_PROGRESS_BAR, so until it rolls out the list has no UI.
      return null;
    case COMPACTION_PART_TYPE: {
      const toolPart = part as ToolUIPart;
      // A failed compaction, or one closed by the abort sentinel (output ""),
      // condensed nothing — settled "Condensed…" copy would report work that
      // never happened. Render nothing; failure messaging belongs to the
      // turn-level error surfaces, not a maintenance row. Same predicate the
      // phase derivation uses, so the row and the bar can never disagree.
      if (isRetiredCompactionRow(part)) return null;
      const settled = toolPart.state === "output-available";
      // A row still open when the stream is over never completed — the
      // user stopped the turn or the connection dropped mid-compaction.
      // Claiming "Condensed the conversation" would report work that
      // never finished, so render nothing, like the abort sentinel.
      if (!isCurrentlyStreaming && !settled) return null;
      const isLiveRow =
        liveCompactionCallId != null &&
        toolPart.toolCallId === liveCompactionCallId;
      const phase = isLiveRow ? (compactionPhase ?? null) : null;
      const outputStats = parseCompactionOutput(
        settled ? toolPart.output : undefined,
      );
      // The streamed `data-compaction` stats pace the live curve while the
      // row is still open; the row's own output wins once it lands.
      const stats = isLiveRow
        ? { ...liveCompactionStats, ...outputStats }
        : outputStats;
      return (
        <CompactionCard
          key={key}
          phase={phase}
          stats={stats}
          // While streaming, an open row with no phase yet is still live —
          // settling on `phase === null` alone would flash the settled copy
          // in the gap before the first progress part lands.
          isSettled={!isCurrentlyStreaming || (settled && phase === null)}
        />
      );
    }
    default:
      // Render a generic tool indicator for SDK built-in
      // tools (Read, Glob, Grep, etc.) or any unrecognized tool
      if (part.type.startsWith("tool-")) {
        return <GenericTool key={key} part={part as ToolUIPart} />;
      }
      return null;
  }
}
