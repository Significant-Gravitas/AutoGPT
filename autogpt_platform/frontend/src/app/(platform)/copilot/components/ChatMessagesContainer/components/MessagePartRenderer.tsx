import { MessageResponse } from "@/components/ai-elements/message";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { StoppedTaskCard } from "./StoppedTaskCard";
import { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { ArtifactCard } from "../../ArtifactCard/ArtifactCard";
import { AskQuestionTool } from "../../../tools/AskQuestion/AskQuestion";
import { ConnectIntegrationTool } from "../../../tools/ConnectIntegrationTool/ConnectIntegrationTool";
import { CreateAgentTool } from "../../../tools/CreateAgent/CreateAgent";
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
import { ViewAgentOutputTool } from "../../../tools/ViewAgentOutput/ViewAgentOutput";
import {
  extractWorkspaceArtifacts,
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

function TextWithArtifactCards({ text }: { text: string }) {
  const isArtifactsEnabled = useGetFlag(Flag.ARTIFACTS);
  const artifacts = extractWorkspaceArtifacts(text);
  const resolved = resolveWorkspaceUrls(text);

  return (
    <>
      {isArtifactsEnabled && artifacts.length > 0 && (
        <div className="mb-2 flex flex-col gap-1">
          {artifacts.map((artifact) => (
            <ArtifactCard key={artifact.id} artifact={artifact} />
          ))}
        </div>
      )}
      <MessageResponse components={STREAMDOWN_COMPONENTS}>
        {resolved}
      </MessageResponse>
    </>
  );
}

interface Props {
  part: UIMessage<unknown, UIDataTypes, UITools>["parts"][number];
  messageID: string;
  partIndex: number;
  onRetry?: () => void;
}

export function MessagePartRenderer({
  part,
  messageID,
  partIndex,
  onRetry,
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

      return <TextWithArtifactCards key={key} text={cleanText} />;
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
    default:
      // Render a generic tool indicator for SDK built-in
      // tools (Read, Glob, Grep, etc.) or any unrecognized tool
      if (part.type.startsWith("tool-")) {
        return <GenericTool key={key} part={part as ToolUIPart} />;
      }
      return null;
  }
}
