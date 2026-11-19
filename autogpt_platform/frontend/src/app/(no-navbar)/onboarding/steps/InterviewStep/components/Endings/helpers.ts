import type { InterviewDetails } from "@/app/api/__generated__/models/interviewDetails";

// The arms of the interview's ending experiment, as their flag variants name
// them, with the plain names testers pick them by.
export const ENDING_NAMES = {
  "02-profile-card": "Profile card",
  "06-coverage-ring": "Coverage ring",
} as const;

export type EndingArm = keyof typeof ENDING_NAMES;
export type EndingMode = "text" | "voice";

// The arms each mode offers, as the server's MODE_ENDINGS does.
export const MODE_ARMS: Record<EndingMode, EndingArm[]> = {
  text: ["02-profile-card", "06-coverage-ring"],
  voice: ["02-profile-card"],
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

// What a start carries: a tester's pick, and the closed interview the
// person asked to start over from, as the ending they saw named it.
export function startBody(
  isPickerOn: boolean,
  mode: EndingMode,
  startOverFrom: string | null,
) {
  const pick = endingPickBody(isPickerOn, mode);
  return startOverFrom
    ? { ...pick, start_over: true, start_over_from: startOverFrom }
    : pick;
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

export const TOPIC_NAMES: Record<string, string> = {
  work: "What you do",
  team: "Your team",
  week: "Where the week goes",
  first_task: "First thing to hand off",
  tools: "Your tools",
};

// --- An ending already settled --------------------------------------------

export interface SettledView {
  heading: string;
  rows: { label: string; value: string }[];
}

// What the profile card's ending shows once the interview is handed off,
// said as a statement; null when none of it can be read.
export function settledViewOf(
  arm: EndingArm,
  details: InterviewDetails | null,
): SettledView | null {
  if (arm !== "02-profile-card") return null;
  const rows = profileRowsOf(details);
  if (rows.length === 0) return null;
  return {
    heading: "Here's what Otto has",
    rows: rows.map(({ label, value }) => ({ label, value })),
  };
}
