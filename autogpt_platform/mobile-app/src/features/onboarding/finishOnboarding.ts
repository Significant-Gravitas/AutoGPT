import { ApiError } from "@/lib/api/client";
import type {
  BrainDumpBody,
  OnboardingHire,
  OnboardingPlan,
  OnboardingProfile,
} from "@/lib/api/onboarding";

import type { WorkAnswer } from "./helpers";

export const PLAN_WAIT_MS = 10_000;
export const PLAN_POLL_MS = 1_000;

/** The calls onboarding ends with, passed in so the order can be tested without a server. */
export interface OnboardingAPI {
  submitProfile(profile: OnboardingProfile): Promise<unknown>;
  finalizeDump(body: BrainDumpBody): Promise<unknown>;
  /** listExperts() answered 402: the server wants a plan first. */
  isPaywalled(): Promise<boolean>;
  getPlan(): Promise<OnboardingPlan>;
  hirePlan(): Promise<OnboardingHire>;
  complete(): Promise<unknown>;
}

export type FinishStage = "saving" | "team" | "finishing";

export type FinishResult =
  | { kind: "done"; notHired: string[] }
  | { kind: "planNeeded" };

interface Args {
  api: OnboardingAPI;
  userName: string;
  role: string;
  work: WorkAnswer;
  recordingID: string;
  onStage(stage: FinishStage, teamNames: string[]): void;
  wait?(ms: number): Promise<void>;
}

/**
 * Saves the answers, hires the team and marks onboarding complete. Throws
 * when an answer can't be saved or the completion fails, to be tried again:
 * the same `recordingID` keeps a second try from saving the dump twice.
 */
export async function finishOnboarding({
  api,
  userName,
  role,
  work,
  recordingID,
  onStage,
  wait = sleep,
}: Args): Promise<FinishResult> {
  onStage("saving", []);
  await api.submitProfile({ user_name: userName, user_role: role, pain_points: [] });
  await api.finalizeDump(
    work.mode === "typed"
      ? { recording_id: recordingID, input_mode: "typed", text: work.text }
      : { recording_id: recordingID, input_mode: "skipped" },
  );

  // No plan yet: the web takes the person through choosing one, and hires then.
  if (await api.isPaywalled()) return { kind: "planNeeded" };

  onStage("team", []);
  const names = await waitForTeam(api, wait, (team) => onStage("team", team));
  let notHired: string[];
  try {
    const hire = await api.hirePlan();
    notHired = hire.failed.map((expert) => expert.name);
  } catch (error) {
    if (isPlanRequired(error)) return { kind: "planNeeded" };
    // Not fatal: Otto takes on their tasks, and experts can be hired later.
    notHired = names;
  }

  onStage("finishing", names);
  await api.complete();
  return { kind: "done", notHired };
}

/** Polls the plan until it is ready or the wait is over; the names are the last answer's team. */
async function waitForTeam(
  api: OnboardingAPI,
  wait: (ms: number) => Promise<void>,
  onTeam: (names: string[]) => void,
) {
  let names: string[] = [];
  // Counted through `wait`, not the clock, so tests need no timers.
  for (let waited = 0; ; waited += PLAN_POLL_MS) {
    let plan: OnboardingPlan;
    try {
      plan = await api.getPlan();
    } catch {
      // The hire settles the plan itself; stop waiting and let it.
      return names;
    }
    names = plan.team.map((expert) => expert.name);
    onTeam(names);
    if (plan.ready || waited >= PLAN_WAIT_MS) return names;
    await wait(PLAN_POLL_MS);
  }
}

export function isPlanRequired(error: unknown) {
  return error instanceof ApiError && error.status === 402;
}

/** "Maria", "Maria and Max", "Maria, Max and Riley". */
export function joinNames(names: string[]) {
  if (names.length <= 1) return names[0] ?? "";
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** The setup screen's list: only what the server is really doing. */
export function checklistFor(work: WorkAnswer, teamNames: string[]) {
  return [
    work.mode === "typed" ? "Reading your brain dump" : "Personalizing your experience",
    teamNames.length > 0 ? `Hiring ${joinNames(teamNames)}` : "Setting up your team",
    "Building your space",
  ];
}

function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}
