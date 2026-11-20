import { ApiError } from "@/lib/autogpt-server-api/helpers";

// "Every Monday at 09:00" reads "every Monday at 09:00" mid-sentence.
export function lowerFirst(text: string): string {
  return text.charAt(0).toLowerCase() + text.slice(1);
}

// Why the API won't touch this chat's cadence, read or change.
export function refusalText(error: unknown): string | null {
  if (!(error instanceof ApiError)) return null;
  if (error.status === 409) return "It's paused. Resume it first.";
  if (error.status === 404)
    return "It no longer runs on a schedule this can change.";
  return null;
}

// The refusals the API names; anything else is ours.
export function changeErrorText(error: unknown): string | null {
  if (error instanceof ApiError && error.status === 400)
    return "Pick one of the options shown.";
  return refusalText(error);
}
