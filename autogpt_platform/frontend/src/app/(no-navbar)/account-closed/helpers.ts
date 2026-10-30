import { formatShortDate } from "@/app/(platform)/settings/billing/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

const FALLBACK_ERROR = "We couldn't keep your account. Try again.";

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
