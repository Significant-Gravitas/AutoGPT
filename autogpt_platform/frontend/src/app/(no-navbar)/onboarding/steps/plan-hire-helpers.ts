import type { hireBrainDumpPlanResponse } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import {
  joinNames,
  pause,
  withDeadline,
} from "./SubscriptionStep/components/PlanPanel/helpers";

// The preparing screen holds its bar this long after the hire goes out, then
// moves on whatever happened: a hire that never answers must not keep anyone
// from Home.
export const HIRE_MAX_WAIT_MS = 15_000;
const HIRE_RETRY_MS = 2_000;
// The proxy waits 30 s for an answer: a try with none by this is sent again
// while another still fits in the bar's wait.
const HIRE_TRY_MS = 7_000;
const HIRE_STARTED_KEY = "autogpt:onboarding-plan-hire:";

// The hire is idempotent, so one the server could not answer (a 5xx, no answer
// within HIRE_TRY_MS, or a plan still changing) is sent again while the
// preparing bar still waits for it.
export async function hireUntilAnswered(
  hirePlan: (signal?: AbortSignal) => Promise<hireBrainDumpPlanResponse>,
  since: number,
) {
  for (;;) {
    // Cut short only while another try still fits: the last one waits for its
    // answer, so a hire that lands after Home still says who joined.
    const isBounded =
      Date.now() - since + HIRE_TRY_MS + HIRE_RETRY_MS < HIRE_MAX_WAIT_MS;
    try {
      const response = isBounded
        ? await withDeadline(hirePlan, HIRE_TRY_MS)
        : await hirePlan();
      if (response.status !== 200) throw new Error("Plan hire failed");
      return response;
    } catch (error) {
      const isRefused =
        error instanceof ApiError && error.status < 500 && error.status !== 409;
      // A bounded try always leaves time for the next, even on a late timer.
      const isOutOfTime =
        !isBounded && Date.now() - since + HIRE_RETRY_MS >= HIRE_MAX_WAIT_MS;
      if (isRefused || isOutOfTime) throw error;
      await pause(HIRE_RETRY_MS);
    }
  }
}

// Marked only while a try is on the wire: a refresh then waits for it
// instead of sending a second, and between tries sends the hire again.
export async function sendMarked(
  userID: string | null,
  hirePlan: () => Promise<hireBrainDumpPlanResponse>,
) {
  markHireStarted(userID);
  try {
    return await hirePlan();
  } finally {
    clearHireStarted(userID);
  }
}

export function toastHireFailed(names: string[]) {
  toast({
    title:
      names.length > 0
        ? `Couldn't hire ${joinNames(names)}`
        : "Couldn't hire your team",
    description:
      "Otto takes on their tasks for now. You can hire experts any time.",
    variant: "destructive",
    // Until closed: the screen moves on to Home, and this should be there.
    duration: Infinity,
  });
}

export function hireTimeLeft(userID: string | null) {
  try {
    const at = Number(sessionStorage.getItem(`${HIRE_STARTED_KEY}${userID}`));
    const ago = Date.now() - at;
    return at > 0 && ago < HIRE_MAX_WAIT_MS ? HIRE_MAX_WAIT_MS - ago : null;
  } catch {
    return null;
  }
}

function markHireStarted(userID: string | null) {
  try {
    sessionStorage.setItem(`${HIRE_STARTED_KEY}${userID}`, String(Date.now()));
  } catch {
    // Without sessionStorage a refresh may send the hire again; it is
    // idempotent, so that costs a request, not a duplicate expert.
  }
}

function clearHireStarted(userID: string | null) {
  try {
    sessionStorage.removeItem(`${HIRE_STARTED_KEY}${userID}`);
  } catch {
    // See markHireStarted.
  }
}
