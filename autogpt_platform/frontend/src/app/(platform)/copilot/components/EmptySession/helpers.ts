import type { User } from "@/lib/auth/types";
import { getExpertRoleLabel as getExpertDisplayRoleLabel } from "@/services/experts/expert-role-label";

export const AUTOPILOT_INTRO =
  "Tell me about your work — I'll find what to automate.";

// A role is free text, and the hire flow's own presets come in both shapes:
// some name a person ("Marketer", "Analyst"), others just a domain ("Sales",
// "Support", "Operations"). "I'm Max, your Sales." reads as an unfinished
// sentence, so a domain gets "expert" appended while a role that already
// names a person is used as it was written.
const PERSON_NOUN_SUFFIXES = ["er", "or", "ist", "yst", "ant", "ian"];
// Person-nouns that none of the suffixes catch.
const PERSON_NOUNS = new Set([
  "agent",
  "assistant",
  "chef",
  "chief",
  "expert",
  "head",
  "lead",
  "pro",
  "rep",
  "specialist",
]);

function namesAPerson(role: string) {
  // Only the last word decides: "Customer Success" is a domain even though
  // "Customer" would pass the suffix test on its own.
  const head = role.split(/\s+/).pop()?.toLowerCase() ?? "";
  if (PERSON_NOUNS.has(head)) return true;
  return PERSON_NOUN_SUFFIXES.some((suffix) => head.endsWith(suffix));
}

export function getExpertRoleLabel(role: string) {
  const displayRole = getExpertDisplayRoleLabel(role);
  return namesAPerson(displayRole) ? displayRole : `${displayRole} expert`;
}

export function getIntroLine(
  expert: { name: string; role: string | null } | null,
) {
  if (!expert) return AUTOPILOT_INTRO;
  return `I'm ${expert.name}${getExpertIntroSuffix(expert.role)}`;
}

export function getExpertIntroSuffix(role: string | null) {
  const trimmedRole = role?.trim();
  return trimmedRole
    ? `, your ${getExpertRoleLabel(trimmedRole)}. What should I take on?`
    : ". What should I take on?";
}

export function getExpertInputPlaceholder(expertName: string) {
  return `What should ${expertName} work on?`;
}

export function getInputPlaceholder(width?: number) {
  if (!width) return "What's your role and what eats up most of your day?";

  if (width < 500) {
    return "I'm a chef and I hate...";
  }
  if (width <= 1080) {
    return "What's your role and what eats up most of your day?";
  }
  return "What's your role and what eats up most of your day? e.g. 'I'm a recruiter and I hate...'";
}

export function getGreetingName(user?: User | null) {
  if (!user) return "there";
  const metadata = user.user_metadata as Record<string, unknown> | undefined;
  // preferred_name is what the user answered to onboarding's "What should I
  // call you?" — it wins over provider-supplied names, and is used verbatim.
  const preferredName = metadata?.preferred_name;
  const fullName = metadata?.full_name;
  const name = metadata?.name;
  if (typeof preferredName === "string" && preferredName.trim()) {
    return preferredName.trim();
  }
  if (typeof fullName === "string" && fullName.trim()) {
    return fullName.split(" ")[0];
  }
  if (typeof name === "string" && name.trim()) {
    return name.split(" ")[0];
  }
  if (user.email) {
    return user.email.split("@")[0];
  }
  return "there";
}
