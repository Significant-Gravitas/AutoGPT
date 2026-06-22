import type { MessagePart } from "../helpers";
import { ReasoningCollapse } from "./ReasoningCollapse";

interface Props {
  parts: MessagePart[];
}

function reasoningText(part: MessagePart): string {
  return "text" in part && typeof part.text === "string" ? part.text : "";
}

export function ReasoningGroup({ parts }: Props) {
  const texts = parts.map(reasoningText).filter((text) => text.trim());
  if (texts.length === 0) return null;

  // The group is still "thinking" while its most recent part streams — that
  // keeps the pulse going through the whole multi-turn reasoning run instead
  // of flickering off between turns.
  const lastPart = parts[parts.length - 1];
  const isActive =
    "state" in lastPart &&
    typeof lastPart.state === "string" &&
    lastPart.state === "streaming";

  return (
    <ReasoningCollapse isActive={isActive}>
      <pre className="whitespace-pre-wrap text-sm text-zinc-700">
        {texts.join("\n\n")}
      </pre>
    </ReasoningCollapse>
  );
}
