import { RAISE_PROMPTS, type RaiseDraft } from "./helpers";

// One beat per question: the question waits for the previous answer, and the
// controls it introduces wait for the question to finish typing.
export const BEAT_KEYS = [
  "role",
  "name",
  "color",
  "avatar",
  "about",
  "voice",
  "budget",
  "marketplace",
  "skills",
] as const;

export type BeatKey = (typeof BEAT_KEYS)[number];

export interface RaiseProgress {
  prompts: Record<BeatKey, boolean>;
  steps: Record<BeatKey, boolean>;
}

export type RaiseFlowItem =
  | { kind: "message"; id: string; role: "autogpt" | "user"; text: string }
  | { kind: "startButton"; id: string }
  | { kind: "step"; id: string; beat: BeatKey };

// Derived from the draft rather than accumulated, so refresh restores and
// back transitions always rebuild the same stream.
export function buildFlowItems(
  draft: RaiseDraft,
  progress: RaiseProgress,
): RaiseFlowItem[] {
  const items: RaiseFlowItem[] = [
    {
      kind: "message",
      id: "autogpt-greeting",
      role: "autogpt",
      text: RAISE_PROMPTS.greeting,
    },
    { kind: "startButton", id: "start-button" },
  ];

  const questions: Record<BeatKey, string> = {
    role: RAISE_PROMPTS.roleQuestion,
    name: RAISE_PROMPTS.nameQuestion,
    color: RAISE_PROMPTS.colorQuestion,
    avatar: RAISE_PROMPTS.avatarQuestion(draft.name),
    about: RAISE_PROMPTS.aboutQuestion(draft.name),
    voice: RAISE_PROMPTS.voiceQuestion(draft.name),
    budget: RAISE_PROMPTS.budgetQuestion(draft.name),
    marketplace: RAISE_PROMPTS.marketplaceQuestion(draft.name),
    skills: RAISE_PROMPTS.skillsQuestion(draft.name),
  };

  BEAT_KEYS.forEach((beat) => {
    if (progress.prompts[beat]) {
      items.push({
        kind: "message",
        id: questionId(beat),
        role: "autogpt",
        text: questions[beat],
      });
    }
    if (progress.steps[beat]) {
      items.push({ kind: "step", id: stepId(beat), beat });
    }
  });

  return items;
}

export function questionId(beat: BeatKey) {
  return `autogpt-${beat}-question`;
}

export function stepId(beat: BeatKey) {
  return `${beat}-step`;
}

// Each beat's question is asked once the beat before it has an answer.
export function beatTriggers(draft: RaiseDraft): Record<BeatKey, boolean> {
  return {
    role: draft.hasStarted,
    name: draft.role !== null,
    color: draft.name !== "",
    avatar: draft.color !== null,
    about: draft.avatarUrl !== null,
    voice: draft.about !== null,
    budget: draft.voiceLabel !== null,
    marketplace: draft.budget !== null,
    skills: draft.marketplace !== null,
  };
}

function beatAnswers(draft: RaiseDraft): Record<BeatKey, boolean> {
  return {
    role: draft.role !== null,
    name: draft.name !== "",
    color: draft.color !== null,
    avatar: draft.avatarUrl !== null,
    about: draft.about !== null,
    voice: draft.voiceLabel !== null,
    budget: draft.budget !== null,
    marketplace: draft.marketplace !== null,
    skills: draft.skills !== null,
  };
}

export function lastAnsweredBeat(draft: RaiseDraft): BeatKey | null {
  const answers = beatAnswers(draft);
  return BEAT_KEYS.filter((beat) => answers[beat]).pop() ?? null;
}

// Going back re-opens a beat by dropping its answer; every question after it
// disappears on its own because the stream is derived from the answers.
export function clearedAnswer(beat: BeatKey): Partial<RaiseDraft> {
  switch (beat) {
    case "role":
      return { role: null };
    case "name":
      return { name: "" };
    case "color":
      return { color: null };
    case "avatar":
      return { avatarUrl: null };
    case "about":
      return { about: null };
    case "voice":
      return { voicePreferences: "", voiceLabel: null };
    case "budget":
      return { budget: null };
    case "marketplace":
      return { marketplace: null };
    case "skills":
      return { skills: null };
  }
}
