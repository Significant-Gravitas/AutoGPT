import type { UIMessage } from "ai";

export interface ScriptedPart {
  part: NonNullable<UIMessage["parts"]>[number];
  delayMs: number;
}

export interface ScriptedTurn {
  assistantMessageId: string;
  /** The prefilled user message for this turn — shown locked in the prompt bar
   * so the visitor only presses Enter to send it. */
  userPrompt: string;
  steps: ScriptedPart[];
}

export type TourScript = ScriptedTurn[];

export interface TourChat {
  id: string;
  title: string;
  updatedAt: string;
  script: TourScript;
}
