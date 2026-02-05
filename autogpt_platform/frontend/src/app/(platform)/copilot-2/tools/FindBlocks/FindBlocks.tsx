import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import type { BlockListResponse } from "@/app/api/__generated__/models/blockListResponse";
import { ToolUIPart } from "ai";
import { getAnimationText, ToolIcon } from "./helpers";

export interface FindBlockInput {
  query: string;
}

export type FindBlockOutput = BlockListResponse;

export interface FindBlockToolPart {
  type: string;
  toolName?: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: FindBlockInput | unknown;
  output?: string | FindBlockOutput | unknown;
  title?: string;
}

interface Props {
  part: FindBlockToolPart;
}

export function FindBlocksTool({ part }: Props) {
  const text = getAnimationText(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError = part.state === "output-error";

  return (
    <div className="flex items-center gap-2 py-2 text-sm text-muted-foreground">
      <ToolIcon isStreaming={isStreaming} isError={isError} />
      <MorphingTextAnimation
        text={text}
        className={isError ? "text-red-500" : undefined}
      />
    </div>
  );
}
