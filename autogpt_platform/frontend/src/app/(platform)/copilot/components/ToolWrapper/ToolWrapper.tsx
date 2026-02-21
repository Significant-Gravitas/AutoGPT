import type { ToolUIPart } from "ai";
import { isLongRunningTool } from "../../tools/long-running-tools";
import { LongRunningToolDisplay } from "../LongRunningToolDisplay/LongRunningToolDisplay";

interface Props {
  part: ToolUIPart;
  children: React.ReactNode;
}

/**
 * Wrapper for all tool components. Automatically shows mini-game
 * for long-running tools by checking LONG_RUNNING_TOOLS list.
 */
export function ToolWrapper({ part, children }: Props) {
  // Extract tool name from part.type (e.g., "tool-create_agent" -> "create_agent")
  const toolName = part.type.replace(/^tool-/, "");
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  return (
    <>
      {/* Automatically show mini-game if tool is long-running and streaming */}
      {isLongRunningTool(toolName) && (
        <LongRunningToolDisplay isStreaming={isStreaming} />
      )}
      {/* Render the actual tool component */}
      {children}
    </>
  );
}
