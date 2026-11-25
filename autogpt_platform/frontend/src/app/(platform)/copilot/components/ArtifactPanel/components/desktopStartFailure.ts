import { ApiError } from "@/lib/autogpt-server-api/helpers";

/** What the desktop-start toast says: why, when the server refused the start. */
export function desktopStartFailure(error: unknown): string {
  if (error instanceof ApiError && error.status === 429) return error.message;
  return "The sandbox did not come up. Try again in a moment.";
}
