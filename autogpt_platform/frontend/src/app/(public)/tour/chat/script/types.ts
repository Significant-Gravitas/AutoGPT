export interface TourPlanStep {
  description: string;
  blockName: string;
}

export interface TourPlan {
  goal: string;
  steps: TourPlanStep[];
}

export interface TourAgent {
  name: string;
  schedule: string;
  blocks: string[];
}

export interface TourArtifact {
  /** Completes the "Artifact · …" caption, e.g. "what lands in your inbox". */
  caption: string;
  title: string;
  subtitle?: string;
  bullets?: string[];
  diff?: { from: string; to: string; delta: string };
}

export type TourPart =
  | { type: "text"; text: string }
  | { type: "plan"; plan: TourPlan }
  | { type: "agent"; agent: TourAgent }
  | { type: "artifact"; artifact: TourArtifact };

export interface TourMessage {
  id: string;
  role: "user" | "assistant";
  parts: TourPart[];
}

export interface ScriptedPart {
  part: TourPart;
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

export interface TourScenario {
  id: string;
  /** Chip label — kept in sync with the product page's scenario chips. */
  label: string;
  script: TourScript;
}
