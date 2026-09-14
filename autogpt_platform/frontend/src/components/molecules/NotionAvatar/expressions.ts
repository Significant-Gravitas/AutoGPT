export type AvatarStatus =
  | "idle"
  | "thinking"
  | "working"
  | "waiting"
  | "done"
  | "failed"
  | "sleeping";

export interface StatusOption {
  id: AvatarStatus;
  label: string;
  hint: string;
}

export const STATUSES: StatusOption[] = [
  { id: "idle", label: "Idle", hint: "calm, slightly curious" },
  { id: "thinking", label: "Thinking", hint: "brows knit, eyes aside" },
  { id: "working", label: "Working", hint: "kicks into gear" },
  { id: "waiting", label: "Needs you", hint: "blocked, asking" },
  { id: "done", label: "Done", hint: "settles, satisfied" },
  { id: "failed", label: "Failed", hint: "falls apart for a beat" },
  { id: "sleeping", label: "Paused", hint: "schedules off, dozing" },
];

export function isAvatarStatus(value: string): value is AvatarStatus {
  return STATUSES.some((status) => status.id === value);
}

/** Overrides on top of the expert's own face. An omitted feature keeps
 *  whatever they were raised with, so a mood never erases their identity. */
export interface Expression {
  eyes?: number;
  eyebrows?: number;
  mouth?: number;
}

// Part indices chosen off the rendered contact sheet: eyes 3 and 11 are drawn
// shut, 9 glances aside, 10 is heavy-lidded; mouth 15 and 18 are flat lines,
// 1 and 10 are smiles.
export const EXPRESSION_POOLS: Record<AvatarStatus, Expression[]> = {
  idle: [{}, {}, { mouth: 0 }, { eyes: 9, mouth: 2 }],
  thinking: [
    { eyes: 9, eyebrows: 14, mouth: 15 },
    { eyes: 9, eyebrows: 9, mouth: 4 },
    { eyes: 4, eyebrows: 3, mouth: 2 },
  ],
  working: [
    { eyes: 10, eyebrows: 10, mouth: 15 },
    { eyes: 10, eyebrows: 4, mouth: 4 },
  ],
  waiting: [
    { eyes: 2, eyebrows: 3, mouth: 5 },
    { eyes: 5, eyebrows: 8, mouth: 11 },
  ],
  done: [
    { eyes: 3, eyebrows: 8, mouth: 1 },
    { eyes: 0, eyebrows: 3, mouth: 10 },
    { eyes: 3, mouth: 3 },
  ],
  failed: [
    { eyes: 12, eyebrows: 15, mouth: 18 },
    { eyes: 8, eyebrows: 5, mouth: 17 },
  ],
  sleeping: [{ eyes: 11, eyebrows: 11, mouth: 16 }],
};

/** Eyes drawn shut — a blink is one frame, not a squash. */
export const BLINK_EYES = 3;

export const EXPRESSION_CADENCE_MS: Record<AvatarStatus, [number, number]> = {
  idle: [7000, 14000],
  thinking: [1800, 3400],
  working: [1800, 3200],
  waiting: [2200, 4200],
  done: [2600, 4800],
  failed: [2400, 4000],
  sleeping: [8000, 12000],
};

export const BLINK_CADENCE_MS: Record<AvatarStatus, [number, number] | null> = {
  idle: [5000, 11000],
  thinking: [3500, 7000],
  working: [2800, 5500],
  waiting: [1800, 3500],
  done: [2500, 5000],
  failed: null,
  sleeping: null,
};
