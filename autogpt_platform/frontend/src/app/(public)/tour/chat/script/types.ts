import type { UIMessage } from "ai";

export interface ScriptedPart {
  part: NonNullable<UIMessage["parts"]>[number];
  delayMs: number;
}

export interface ScriptedTurn {
  assistantMessageId: string;
  steps: ScriptedPart[];
}

export type TourScript = ScriptedTurn[];
