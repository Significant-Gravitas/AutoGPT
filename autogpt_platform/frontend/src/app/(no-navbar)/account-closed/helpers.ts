import type { AccountDeletionResponseStatus } from "@/app/api/__generated__/models/accountDeletionResponseStatus";
import { formatShortDate } from "@/app/(platform)/settings/billing/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

const FALLBACK_ERROR = "We couldn't keep your account. Try again.";

// How the page waits, after Keep my account, for every server to stop
// answering the account as closed: each caches that answer for up to 30 s.
export const OPEN_PROBE = {
  intervalMS: 3000,
  // Several in a row, as each request may reach a different server.
  streak: 3,
  // Longer than any server's cache, counted from the keep.
  minWaitMS: 35_000,
  // About two minutes; then Continue shows anyway, next to the help line.
  maxProbes: 40,
};

export type PageView =
  | "loading"
  | "error"
  | "closed"
  | "too_late"
  | "open"
  | "leaving";

interface PageViewArgs {
  isKept: boolean;
  isTooLate: boolean;
  isLoading: boolean;
  isError: boolean;
  status: AccountDeletionResponseStatus | undefined;
}

export function getPageView({
  isKept,
  isTooLate,
  isLoading,
  isError,
  status,
}: PageViewArgs): PageView {
  if (isKept) return "open";
  // The server decides: a 409 on Keep, or its too_late status once the date
  // has passed or the erasure has started. Never the device clock.
  if (isTooLate || status === "too_late") return "too_late";
  // Not kept on this page: never closed, or kept earlier (a reload, another
  // tab). Back to the app, once ordinary requests answer it as open.
  if (status === "open") return "leaving";
  if (isLoading) return "loading";
  if (isError || !status) return "error";
  return "closed";
}

export function getEraseDateLabel(
  eraseAfter: Date | string | null | undefined,
) {
  return eraseAfter ? formatShortDate(eraseAfter) : null;
}

export function isTooLateError(error: unknown) {
  return error instanceof ApiError && error.status === 409;
}

export function getKeepErrorMessage(error: unknown): string {
  return error instanceof ApiError && error.message
    ? error.message
    : FALLBACK_ERROR;
}

interface WaitArgs {
  probe: () => Promise<unknown>;
  isCancelled: () => boolean;
  settings: typeof OPEN_PROBE;
  keptAt: number;
  now?: () => number;
}

// Probes an ordinary route until it answers the account as open several
// times in a row and every server's cache has run out since the keep, or
// gives up after settings.maxProbes.
export async function waitUntilAnsweredAsOpen({
  probe,
  isCancelled,
  settings,
  keptAt,
  now = Date.now,
}: WaitArgs) {
  let streak = 0;
  for (let attempt = 1; attempt <= settings.maxProbes; attempt++) {
    if (isCancelled()) return;
    streak = (await answersAsOpen(probe)) ? streak + 1 : 0;
    const waitedLongEnough = now() - keptAt >= settings.minWaitMS;
    if (streak >= settings.streak && waitedLongEnough) return;
    if (attempt === settings.maxProbes) return;
    await wait(settings.intervalMS);
  }
}

// Only an answer that proves a signed-in, open account counts: a success, or
// a 4xx refusal other than 401 (which a closed account and a lost session
// both get). A 5xx or a network failure says nothing.
async function answersAsOpen(probe: () => Promise<unknown>) {
  try {
    await probe();
    return true;
  } catch (error) {
    if (!(error instanceof ApiError)) return false;
    return error.status >= 400 && error.status < 500 && error.status !== 401;
  }
}

function wait(ms: number): Promise<void> {
  return new Promise(function resolveAfterDelay(resolve) {
    setTimeout(resolve, ms);
  });
}
