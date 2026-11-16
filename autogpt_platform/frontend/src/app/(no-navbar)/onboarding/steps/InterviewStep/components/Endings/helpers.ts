import type { InterviewDetails } from "@/app/api/__generated__/models/interviewDetails";

// The arms of the interview's ending experiment, as their flag variants name
// them, with the plain names testers pick them by.
export const ENDING_NAMES = {
  "02-profile-card": "Profile card",
  "03-summary-chips": "Summary chips",
  "04-team-reveal": "Team reveal",
  "06-coverage-ring": "Coverage ring",
  "08-transcript-highlights": "Transcript highlights",
  "09-first-task": "First task",
} as const;

export type EndingArm = keyof typeof ENDING_NAMES;
export type EndingMode = "text" | "voice";

// The arms each mode offers, as the server's MODE_ENDINGS does.
export const MODE_ARMS: Record<EndingMode, EndingArm[]> = {
  text: [
    "02-profile-card",
    "03-summary-chips",
    "04-team-reveal",
    "06-coverage-ring",
    "08-transcript-highlights",
    "09-first-task",
  ],
  voice: [
    "02-profile-card",
    "03-summary-chips",
    "04-team-reveal",
    "08-transcript-highlights",
    "09-first-task",
  ],
};

export function endingArmOf(value: unknown): EndingArm | null {
  return typeof value === "string" && value in ENDING_NAMES
    ? (value as EndingArm)
    : null;
}

// --- A tester's pick ------------------------------------------------------

const PICK_KEY_PREFIX = "autogpt:onboarding-ending-pick:";

export function readEndingPick(mode: EndingMode): EndingArm | null {
  if (typeof window === "undefined") return null;
  try {
    const arm = endingArmOf(localStorage.getItem(`${PICK_KEY_PREFIX}${mode}`));
    return arm && MODE_ARMS[mode].includes(arm) ? arm : null;
  } catch {
    return null;
  }
}

export function saveEndingPick(mode: EndingMode, arm: EndingArm | null) {
  try {
    const key = `${PICK_KEY_PREFIX}${mode}`;
    if (arm) localStorage.setItem(key, arm);
    else localStorage.removeItem(key);
  } catch {
    // The pick lasts for this page only.
  }
}

// The pick a start carries: none unless the picker is on and something
// other than "Default" is picked.
export function endingPickBody(isPickerOn: boolean, mode: EndingMode) {
  const arm = isPickerOn ? readEndingPick(mode) : null;
  return arm ? { ending_pick: arm } : undefined;
}

// --- Corrections, in the person's voice -----------------------------------

const MAX_CORRECTION_CHARS = 2_000;

// What the person changed on an ending screen, as the hand-off takes it:
// one line each, null when they changed nothing.
export function correctionsOf(lines: (string | null | undefined)[]) {
  const kept = lines
    .map((line) => line?.trim())
    .filter((line): line is string => !!line);
  if (kept.length === 0) return null;
  return kept.join("\n").slice(0, MAX_CORRECTION_CHARS);
}

export function firstTaskCorrection(original: string, edited: string) {
  const task = edited.trim();
  if (!task || task === original.trim()) return null;
  return `My first task should be: ${task}`;
}

export function notRightCorrection(item: string) {
  return `Not right: ${item}`;
}

export function alsoCorrection(item: string) {
  return `Also: ${item}`;
}

// --- What the interview heard ---------------------------------------------

export interface ProfileRow {
  key: "work" | "team" | "time" | "tools";
  label: string;
  value: string;
}

function joined(items: (string | null | undefined)[]) {
  return items.filter((item): item is string => !!item?.trim()).join(", ");
}

// The profile card's rows, each only once something fills it.
export function profileRowsOf(details: InterviewDetails | null): ProfileRow[] {
  if (!details) return [];
  const rows: ProfileRow[] = [
    {
      key: "work",
      label: "Your work",
      value: joined([details.business, details.role]),
    },
    { key: "team", label: "Your team", value: details.team ?? "" },
    {
      key: "time",
      label: "What eats your time",
      value: joined(details.time_sinks ?? []),
    },
    { key: "tools", label: "Your tools", value: joined(details.tools ?? []) },
  ];
  return rows.filter((row) => row.value);
}

export function profileCorrections(rows: ProfileRow[], edited: ProfileRow[]) {
  return correctionsOf(
    edited.flatMap((row, index) => {
      const before = rows[index];
      if (!before || before.value === row.value.trim()) return [];
      return [
        notRightCorrection(`${before.label}: ${before.value}`),
        row.value.trim() ? alsoCorrection(`${row.label}: ${row.value}`) : null,
      ];
    }),
  );
}

// "Here's what I heard": one chip per thing heard, never the same twice.
export function chipsOf(details: InterviewDetails | null): string[] {
  if (!details) return [];
  const chips = [
    details.business,
    details.role,
    details.team,
    ...(details.time_sinks ?? []),
    ...(details.tools ?? []),
    details.first_task,
  ].filter((chip): chip is string => !!chip?.trim());
  return Array.from(new Set(chips));
}

export function chipCorrections(struck: string[], added: string) {
  return correctionsOf([
    ...struck.map(notRightCorrection),
    added.trim() ? alsoCorrection(added) : null,
  ]);
}

export interface Segment {
  text: string;
  // The fact this part of the line says, when it is one.
  fact: number | null;
}

// A line of the person's own words, cut where a fact's quote falls in it,
// case aside. A quote found in no line marks nothing.
export function highlightSegments(line: string, quotes: string[]): Segment[] {
  const marks: { start: number; end: number; fact: number }[] = [];
  const lower = line.toLowerCase();
  quotes.forEach((quote, fact) => {
    const needle = quote.trim().toLowerCase();
    if (!needle) return;
    const start = lower.indexOf(needle);
    if (start < 0) return;
    const end = start + needle.length;
    if (marks.some((mark) => start < mark.end && end > mark.start)) return;
    marks.push({ start, end, fact });
  });
  marks.sort((a, b) => a.start - b.start);
  const segments: Segment[] = [];
  let at = 0;
  for (const mark of marks) {
    if (mark.start > at)
      segments.push({ text: line.slice(at, mark.start), fact: null });
    segments.push({ text: line.slice(mark.start, mark.end), fact: mark.fact });
    at = mark.end;
  }
  if (at < line.length) segments.push({ text: line.slice(at), fact: null });
  return segments;
}

export const TOPIC_NAMES: Record<string, string> = {
  work: "What you do",
  team: "Your team",
  week: "Where the week goes",
  first_task: "First thing to hand off",
  tools: "Your tools",
};
