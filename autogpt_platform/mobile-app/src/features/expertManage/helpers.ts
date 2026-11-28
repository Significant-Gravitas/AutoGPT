import type { AutopilotMode, ExpertSoul } from "../../lib/api/expertManage";

const CREDITS_PER_DOLLAR = 100;
// The server's caps (`backend/api/features/experts/models.py`).
const MAX_BUDGET_CREDITS = 1_000_000;
const NAME_MAX = 100;
const IDENTITY_MAX = 10_000;
const SECTION_MAX = 4_000;

// Expert ids are UUIDs (`@default(uuid())`); the route param can come from a link.
const EXPERT_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isExpertID(value: unknown): value is string {
  return typeof value === "string" && EXPERT_ID.test(value);
}

const DOLLARS = /^\d+(\.\d{1,2})?$/;
const BUDGET_ERROR = "Enter a dollar amount like 25 or 12.50";

/** A typed dollar amount as credits; empty means the platform default. */
export function parseBudget(text: string): { credits: number | null } | { error: string } {
  const trimmed = text.trim();
  if (!trimmed) return { credits: null };
  if (!DOLLARS.test(trimmed)) return { error: BUDGET_ERROR };
  const credits = Math.round(Number(trimmed) * CREDITS_PER_DOLLAR);
  if (credits > MAX_BUDGET_CREDITS) return { error: BUDGET_ERROR };
  return { credits };
}

/** The effective budget as the field starts out. */
export function budgetInput(credits: number | null) {
  if (credits === null) return "0";
  const dollars = credits / CREDITS_PER_DOLLAR;
  return Number.isInteger(dollars) ? String(dollars) : dollars.toFixed(2);
}

export function formatDollars(credits: number) {
  const dollars = credits / CREDITS_PER_DOLLAR;
  return Number.isInteger(dollars)
    ? `$${dollars.toLocaleString("en-US")}`
    : `$${dollars.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/** The server returns null when there is no cap. */
export function formatBudget(credits: number | null) {
  if (credits === null) return "No limit";
  return formatDollars(credits);
}

export function validateProfile(form: ExpertSoul) {
  const name = form.name.trim();
  if (!name) return "Give your expert a name.";
  if (name.length > NAME_MAX) return "Keep the name to 100 characters or fewer.";
  const identity = form.identity.trim();
  if (!identity) return "Say who this expert is.";
  if (identity.length > IDENTITY_MAX) return "Keep the identity under 10,000 characters.";
  if (
    form.voice_preferences.trim().length > SECTION_MAX ||
    form.boundaries.trim().length > SECTION_MAX
  ) {
    return "Keep each section to 4,000 characters or fewer.";
  }
  return null;
}

// The server trims every field, so only a change past the spaces is a change.
export function isProfileDirty(saved: ExpertSoul, form: ExpertSoul) {
  return (Object.keys(saved) as (keyof ExpertSoul)[]).some(
    (field) => saved[field].trim() !== form[field].trim(),
  );
}

export function trimProfile(form: ExpertSoul): ExpertSoul {
  return {
    name: form.name.trim(),
    identity: form.identity.trim(),
    voice_preferences: form.voice_preferences.trim(),
    boundaries: form.boundaries.trim(),
  };
}

interface ModeOption {
  value: AutopilotMode;
  label: string;
  description: string;
}

// Word for word from the web's AutopilotModeSelector.
export const MODE_OPTIONS: ModeOption[] = [
  {
    value: "ask_first",
    label: "Ask first",
    description:
      "Asks before every edit or outside action. Work that stays in its sandbox runs.",
  },
  {
    value: "auto",
    label: "Auto",
    description:
      "Asks before anything leaves the platform, and about edits or commands that look risky.",
  },
  {
    value: "unsupervised",
    label: "Unsupervised",
    description: "Never asks. Every action runs without your approval.",
  },
];

/** The warning before an expert's new chats stop asking. */
export function unsupervisedPrompt(name: string) {
  return {
    title: `Run ${name} unsupervised by default?`,
    body: `${name} will not ask before edits, commands or outside actions in new chats. You can change the default here or switch modes for a single chat on the web.`,
  };
}
