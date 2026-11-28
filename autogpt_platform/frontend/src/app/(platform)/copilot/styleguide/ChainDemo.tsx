import type { ToolUIPart } from "ai";
import type { MessagePart } from "../components/ChatMessagesContainer/helpers";
import { ToolChain } from "../components/ToolChain/ToolChain";

interface Props {
  part: {
    type: `tool-${string}`;
    toolCallId: string;
    state: ToolUIPart["state"];
    input?: unknown;
    output?: unknown;
    errorText?: string;
  };
}

/** A tool call as the chat draws it: one row of the tool chain. */
export function ChainDemo({ part }: Props) {
  const running =
    part.state === "input-streaming" || part.state === "input-available";
  return (
    <ToolChain parts={[part as unknown as MessagePart]} isStreaming={running} />
  );
}
