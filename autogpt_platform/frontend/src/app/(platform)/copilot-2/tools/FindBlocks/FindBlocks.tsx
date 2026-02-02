import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { ToolUIPart } from "ai";
import { getAnimationText, StateIcon } from "./helpers";

export interface FindBlockInput {
  query: string;
}

export interface FindBlockOutput {
  type: "block_list";
  message: string;
  session_id: string;
  blocks: BlockInfo[];
  count: number;
  query: string;
  usage_hint: string;
}

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

  return (
    <div className="flex items-center gap-2 py-2 text-sm text-muted-foreground">
      <StateIcon state={part.state} />
      <MorphingTextAnimation text={text} />
    </div>
  );
}
