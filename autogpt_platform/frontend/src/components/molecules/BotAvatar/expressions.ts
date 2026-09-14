import type { AvatarStatus } from "./helpers";

export type ExpressionId =
  | "neutral"
  | "soft"
  | "wide"
  | "big"
  | "small"
  | "sleepy"
  | "wink"
  | "angry"
  | "happy"
  | "sad"
  | "suspicious"
  | "curious"
  | "dots"
  | "aside"
  | "focused"
  | "cross";

export type EyeKind = "open" | "arc" | "flat" | "cross";
export type BrowKind = "none" | "raised" | "wave" | "furrow" | "sad" | "flat";

export interface EyeSpec {
  kind: EyeKind;
  rx: number;
  ry: number;
  tilt: number;
  dx: number;
  dy: number;
}

export interface ExpressionSpec {
  id: ExpressionId;
  label: string;
  left: EyeSpec;
  right: EyeSpec;
  brow: BrowKind;
  mouth: string;
  blush: number;
}

const OPEN: EyeSpec = { kind: "open", rx: 4.5, ry: 6, tilt: 0, dx: 0, dy: 0 };

function eye(patch: Partial<EyeSpec>): EyeSpec {
  return { ...OPEN, ...patch };
}

const MOUTH = {
  smile: "M-4,0 Q0,3.5 4,0",
  grin: "M-6,-0.5 Q0,6.5 6,-0.5",
  flat: "M-3.5,0.5 L3.5,0.5",
  o: "M-2.4,0 A2.4,2.4 0 1 0 2.4,0 A2.4,2.4 0 1 0 -2.4,0",
  dot: "M-1.8,0 A1.8,1.8 0 1 0 1.8,0 A1.8,1.8 0 1 0 -1.8,0",
  frown: "M-4,1.5 Q0,-2 4,1.5",
  wobble: "M-5,1 Q-2.5,-2 0,1 Q2.5,4 5,1",
  smirk: "M-3,0.5 Q0,2 3,-0.5",
  tiny: "M-2,0.5 L2,0.5",
};

export const EXPRESSIONS: ExpressionSpec[] = [
  {
    id: "neutral",
    label: "Neutral",
    left: OPEN,
    right: OPEN,
    brow: "none",
    mouth: MOUTH.smile,
    blush: 0.5,
  },
  {
    id: "soft",
    label: "Soft",
    left: eye({ ry: 5.2 }),
    right: eye({ ry: 5.2 }),
    brow: "none",
    mouth: MOUTH.smirk,
    blush: 0.55,
  },
  {
    id: "wide",
    label: "Wide",
    left: eye({ rx: 5.4, ry: 7.2, dy: -0.5 }),
    right: eye({ rx: 5.4, ry: 7.2, dy: -0.5 }),
    brow: "raised",
    mouth: MOUTH.o,
    blush: 0.5,
  },
  {
    id: "big",
    label: "Big",
    left: eye({ rx: 5, ry: 8.5, dy: -0.8 }),
    right: eye({ rx: 5, ry: 8.5, dy: -0.8 }),
    brow: "raised",
    mouth: MOUTH.grin,
    blush: 0.75,
  },
  {
    id: "small",
    label: "Small",
    left: eye({ rx: 3.4, ry: 4.2, dy: 0.6 }),
    right: eye({ rx: 3.4, ry: 4.2, dy: 0.6 }),
    brow: "none",
    mouth: MOUTH.tiny,
    blush: 0.85,
  },
  {
    id: "sleepy",
    label: "Sleepy",
    left: eye({ kind: "flat", rx: 5, ry: 1.6 }),
    right: eye({ kind: "flat", rx: 5, ry: 1.6 }),
    brow: "none",
    mouth: MOUTH.dot,
    blush: 0.4,
  },
  {
    id: "wink",
    label: "Wink",
    left: OPEN,
    right: eye({ kind: "arc" }),
    brow: "none",
    mouth: MOUTH.smirk,
    blush: 0.6,
  },
  {
    id: "angry",
    label: "Angry",
    left: eye({ ry: 4.6, tilt: -18, dy: 0.6 }),
    right: eye({ ry: 4.6, tilt: 18, dy: 0.6 }),
    brow: "furrow",
    mouth: MOUTH.flat,
    blush: 0.4,
  },
  {
    id: "happy",
    label: "Happy",
    left: eye({ kind: "arc" }),
    right: eye({ kind: "arc" }),
    brow: "none",
    mouth: MOUTH.grin,
    blush: 0.85,
  },
  {
    id: "sad",
    label: "Sad",
    left: eye({ ry: 5.4, tilt: 16, dy: 0.8 }),
    right: eye({ ry: 5.4, tilt: -16, dy: 0.8 }),
    brow: "sad",
    mouth: MOUTH.frown,
    blush: 0.35,
  },
  {
    id: "suspicious",
    label: "Suspicious",
    left: eye({ ry: 4.4, dy: 0.5 }),
    right: eye({ kind: "flat", rx: 5, ry: 1.8, dy: 0.5 }),
    brow: "flat",
    mouth: MOUTH.smirk,
    blush: 0.4,
  },
  {
    id: "curious",
    label: "Curious",
    left: eye({ rx: 5.4, ry: 7.4, dy: -0.6 }),
    right: eye({ rx: 3.8, ry: 4.8, dy: 0.4 }),
    brow: "raised",
    mouth: MOUTH.smirk,
    blush: 0.5,
  },
  {
    id: "dots",
    label: "Dots",
    left: eye({ rx: 3.2, ry: 3.2, dx: 1.5, dy: -2 }),
    right: eye({ rx: 3.2, ry: 3.2, dx: 1.5, dy: -2 }),
    brow: "wave",
    mouth: MOUTH.smirk,
    blush: 0.45,
  },
  {
    id: "aside",
    label: "Aside",
    left: eye({ dx: 2.6, dy: -0.8 }),
    right: eye({ dx: 2.6, dy: -0.8 }),
    brow: "none",
    mouth: MOUTH.tiny,
    blush: 0.5,
  },
  {
    id: "focused",
    label: "Focused",
    left: eye({ ry: 4.2, dx: 0.6, dy: 1 }),
    right: eye({ ry: 4.2, dx: 0.6, dy: 1 }),
    brow: "furrow",
    mouth: MOUTH.flat,
    blush: 0.45,
  },
  {
    id: "cross",
    label: "Cross",
    left: eye({ kind: "cross" }),
    right: eye({ kind: "cross" }),
    brow: "none",
    mouth: MOUTH.wobble,
    blush: 0.35,
  },
];

export const POOLS: Record<AvatarStatus, ExpressionId[]> = {
  idle: ["neutral", "soft", "aside", "small"],
  thinking: ["dots", "aside", "curious", "suspicious"],
  working: ["focused", "neutral", "focused", "soft"],
  waiting: ["wide", "curious", "big", "sad"],
  done: ["happy", "big", "wink"],
  failed: ["cross", "sad", "angry"],
  sleeping: ["sleepy"],
};

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

export function findExpression(id: ExpressionId) {
  return (
    EXPRESSIONS.find((expression) => expression.id === id) ?? EXPRESSIONS[0]
  );
}

export function isExpressionId(value: string): value is ExpressionId {
  return EXPRESSIONS.some((expression) => expression.id === value);
}
