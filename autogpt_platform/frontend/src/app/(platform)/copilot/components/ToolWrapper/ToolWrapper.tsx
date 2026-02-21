import type { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { LongRunningToolDisplay } from "../LongRunningToolDisplay/LongRunningToolDisplay";

interface Props {
  part: ToolUIPart;
  message: UIMessage<unknown, UIDataTypes, UITools>;
  children: React.ReactNode;
}

/**
 * Wrapper for all tool components. Automatically shows UI feedback (e.g., mini-game)
 * for long-running tools by detecting StreamLongRunningStart events from the backend.
 */
export function ToolWrapper({ part, message, children }: Props) {
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  // Check if this tool has a long-running-start event in the message
  const isLongRunning = message.parts.some(
    (p) =>
      p.type === "long-running-start" &&
      "toolCallId" in p &&
      "toolCallId" in part &&
      p.toolCallId === part.toolCallId,
  );

  return (
    <>
      {/* Show UI feedback if tool is long-running and streaming */}
      {isLongRunning && <LongRunningToolDisplay isStreaming={isStreaming} />}
      {/* Render the actual tool component */}
      {children}
    </>
  );
}
