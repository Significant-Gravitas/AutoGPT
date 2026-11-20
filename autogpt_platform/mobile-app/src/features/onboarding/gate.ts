// No native imports, so tests can load this without a phone.
import { serverSlug } from "@/lib/auth/sessionHelpers";

export type OnboardingGate = "loading" | "needed" | "done";

/** How long launch waits for the server before letting the person in. */
export const GATE_TIMEOUT_MS = 8_000;

/** Where the phone remembers that one account on one server finished onboarding. */
export function onboardedKey(serverURL: string, userID: string) {
  // SecureStore keys take letters, digits, ".", "-" and "_" only.
  const user = userID.replace(/[^A-Za-z0-9._-]/g, "_");
  return `autogpt.onboarded.${serverSlug(serverURL)}.${user}`;
}

/**
 * Whether a signed-in account sees onboarding. A server that can't be asked
 * lets the person in: an existing account must not sit on a spinner.
 */
export function gateState({
  isRemembered,
  status,
  isCompleted,
}: {
  isRemembered: boolean;
  status: "pending" | "error" | "success";
  isCompleted: boolean | undefined;
}): OnboardingGate {
  if (isRemembered) return "done";
  if (status === "pending") return "loading";
  if (status === "error") return "done";
  return isCompleted ? "done" : "needed";
}
