import type { ToolUIPart } from "ai";
import { LongRunningToolDisplay } from "../LongRunningToolDisplay/LongRunningToolDisplay";

interface Props {
  part: ToolUIPart;
  children: React.ReactNode;
}

/**
 * Wrapper for all tool components. Automatically shows UI feedback
 * for long-running tools by detecting the isLongRunning flag on the tool part.
 */
export function ToolWrapper({ part, children }: Props) {
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  // Check if this tool is marked as long-running via providerMetadata
  const isLongRunning =
    "providerMetadata" in part &&
    part.providerMetadata &&
    typeof part.providerMetadata === "object" &&
    "isLongRunning" in part.providerMetadata &&
    part.providerMetadata.isLongRunning === true;

  // Debug logging
  if (part.type.startsWith("tool-")) {
    console.log("[ToolWrapper]", {
      toolName: "toolName" in part ? part.toolName : "unknown",
      hasProviderMetadata: "providerMetadata" in part,
      providerMetadata:
        "providerMetadata" in part ? part.providerMetadata : undefined,
      computed: isLongRunning,
      state: part.state,
      part,
    });
  }

  return (
    <>
      {/* Show UI feedback if tool is long-running and streaming */}
      {isLongRunning && <LongRunningToolDisplay isStreaming={isStreaming} />}
      {/* Render the actual tool component */}
      {children}
    </>
  );
}
