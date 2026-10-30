import type { AccountDeletionResponseStatus } from "@/app/api/__generated__/models/accountDeletionResponseStatus";
import { formatShortDate } from "@/app/(platform)/settings/billing/helpers";
import { isAccountClosedError } from "@/lib/auth/closed-account";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

const FALLBACK_ERROR = "We couldn't keep your account. Try again.";

// How the page waits, after Keep my account, for the other servers to stop
// answering the account as closed: they cache that answer for up to 30 s.
export const OPEN_PROBE = {
  intervalMS: 3000,
  // Several in a row, as each request may reach a different server.
  streak: 3,
  // About two minutes; then Continue shows anyway, next to the help line.
  maxProbes: 40,
};

export type PageView = "loading" | "error" | "closed" | "too_late" | "open";

interface PageViewArgs {
  isKept: boolean;
  isLoading: boolean;
  isError: boolean;
  status: AccountDeletionResponseStatus | undefined;
  eraseAfter: Date | string | null | undefined;
  now?: Date;
}

export function getPageView({
  isKept,
  isLoading,
  isError,
  status,
  eraseAfter,
  now = new Date(),
}: PageViewArgs): PageView {
  if (isKept || status === "open") return "open";
  if (isLoading) return "loading";
  if (isError || !status) return "error";
  // Past the erasure date the account can't be kept, even before the
  // erasure job has run.
  if (eraseAfter && new Date(eraseAfter).getTime() <= now.getTime()) {
    return "too_late";
  }
  return "closed";
}

export function getEraseDateLabel(
  eraseAfter: Date | string | null | undefined,
) {
  return eraseAfter ? formatShortDate(eraseAfter) : null;
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
}

// Probes an ordinary route until it stops answering the account as closed
// several times in a row, or gives up after settings.maxProbes.
export async function waitUntilAnsweredAsOpen({
  probe,
  isCancelled,
  settings,
}: WaitArgs) {
  let streak = 0;
  for (let attempt = 0; attempt < settings.maxProbes; attempt++) {
    if (isCancelled()) return;
    streak = (await answersAsOpen(probe)) ? streak + 1 : 0;
    if (streak >= settings.streak) return;
    await wait(settings.intervalMS);
  }
}

async function answersAsOpen(probe: () => Promise<unknown>) {
  try {
    await probe();
    return true;
  } catch (error) {
    // Any other refusal still means this server sees the account as open.
    return error instanceof ApiError && !isAccountClosedError(error);
  }
}

function wait(ms: number): Promise<void> {
  return new Promise(function resolveAfterDelay(resolve) {
    setTimeout(resolve, ms);
  });
}
