import type { PlanProgress } from "@/app/api/__generated__/models/planProgress";
import type { PlanResponse } from "@/app/api/__generated__/models/planResponse";
import type { RecommendedExpert } from "@/app/api/__generated__/models/recommendedExpert";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import type { ChecklistState } from "../../../../components/ChecklistMark";

export const PLAN_POLL_MS = 1_000;
// The pipeline is done long before this on a good day. Past it the step stops
// waiting and the role plan the last answer carried becomes the plan, for the
// trial and Home too, so the cut-off is generous: what it costs is the plan
// the brain dump would have written.
export const PLAN_MAX_WAIT_MS = 30_000;
export const PLAN_STEP_TASKS = 3;

// The labels RoleStep shows, and what to call someone in that role.
const ROLE_COPY: Record<string, { label: string; noun: string }> = {
  "Founder/CEO": { label: "Founder / CEO", noun: "founder" },
  Operations: { label: "Operations", noun: "operations lead" },
  "Sales/BD": { label: "Sales / BD", noun: "sales lead" },
  Marketing: { label: "Marketing", noun: "marketer" },
  "Product/PM": { label: "Product / PM", noun: "product manager" },
  Engineering: { label: "Engineering", noun: "engineer" },
  "HR/People": { label: "HR / People", noun: "people lead" },
};

const SHARPEN_BUSINESS =
  "Tell me about your business once you're in and I'll sharpen it.";
const SHARPEN_WORK =
  "Tell me about your work once you're in and I'll sharpen it.";

export type ConsentMoment = "trial" | "subscribe" | "continue";

const JOIN_MOMENT: Record<ConsentMoment, string> = {
  trial: "when your trial starts",
  subscribe: "when you subscribe",
  continue: "when you continue",
};

export const NOTHING_RUNS = "Nothing runs until you pick a task.";

interface RoleHeardArgs {
  role: string;
  skippedDump: boolean;
}

// What the "Based on your role" card says. Only a skipped dump is called a
// skip: a dump that failed or is still being read was not skipped.
export function roleHeard({ role, skippedDump }: RoleHeardArgs) {
  const copy = ROLE_COPY[role];
  const skipped = skippedDump ? "You skipped the brain dump, so this" : "This";
  if (!copy) return `${skipped} is a first plan to get going. ${SHARPEN_WORK}`;
  const plan = `${skipped} is a first plan for ${withArticle(copy.noun)}.`;
  return `${copy.label}. ${plan} ${SHARPEN_BUSINESS}`;
}

function withArticle(noun: string) {
  return /^[aeiou]/i.test(noun) ? `an ${noun}` : `a ${noun}`;
}

export function joinNames(names: string[]) {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function consentLine(names: string[], moment: ConsentMoment) {
  if (names.length === 0)
    return `Otto takes these on for now. You can hire experts any time. ${NOTHING_RUNS}`;
  const verb = names.length === 1 ? "joins" : "join";
  return `${joinNames(names)} ${verb} your team ${JOIN_MOMENT[moment]}. ${NOTHING_RUNS}`;
}

export function getTaskOwner(
  task: SuggestedTask,
  team: RecommendedExpert[],
): RecommendedExpert | null {
  if (!task.expert_template_id) return null;
  return (
    team.find((expert) => expert.template_id === task.expert_template_id) ??
    null
  );
}

export function getStepTasks(plan: PlanResponse) {
  return (plan.tasks ?? []).slice(0, PLAN_STEP_TASKS);
}

// What the server is doing while the step waits: the tasks and the team are
// worked out side by side, then each task gets its owner.
const PROGRESS_STEPS = [
  "Turning your brain dump into tasks",
  "Finding experts who fit your work",
  "Choosing who takes each task",
] as const;

export function getProgressSteps(progress: PlanProgress) {
  const written = Boolean(progress.tasks_written);
  const chosen = Boolean(progress.team_chosen);
  const states: ChecklistState[] = [
    written ? "done" : "active",
    chosen ? "done" : "active",
    written && chosen ? "active" : "pending",
  ];
  return PROGRESS_STEPS.map((label, index) => ({
    label,
    state: states[index],
  }));
}

export const EMPTY_PLAN: PlanResponse = {
  ready: false,
  source: "fallback",
  skipped_dump: false,
  heard: null,
  team: [],
  tasks: [],
};

const SHOWN_PLAN_KEY = "autogpt:onboarding-plan:";

// Stripe sends an abandoned checkout back here; the step should come back as
// it was left, not as skeletons.
export function isCheckoutReturn(params: URLSearchParams) {
  return (
    params.get("subscription") === "cancelled" ||
    params.get("trial") === "cancelled"
  );
}

export function rememberShownPlan(userID: string | null, plan: PlanResponse) {
  if (!userID) return;
  try {
    sessionStorage.setItem(`${SHOWN_PLAN_KEY}${userID}`, JSON.stringify(plan));
  } catch {
    // In-app browsers may block sessionStorage; the return then refetches.
  }
}

// A new take or a skip: the plan shown for the last take is not this one's.
export function forgetShownPlan(userID: string | null) {
  if (!userID) return;
  try {
    sessionStorage.removeItem(`${SHOWN_PLAN_KEY}${userID}`);
  } catch {
    // Nothing was stored, then.
  }
}

export function readShownPlan(userID: string | null): PlanResponse | null {
  if (!userID) return null;
  try {
    const plan = JSON.parse(
      sessionStorage.getItem(`${SHOWN_PLAN_KEY}${userID}`) ?? "null",
    );
    return plan && Array.isArray(plan.tasks) && Array.isArray(plan.team)
      ? (plan as PlanResponse)
      : null;
  } catch {
    return null;
  }
}
