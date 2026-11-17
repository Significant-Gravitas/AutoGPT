import { createHash } from "node:crypto";
import { type AuthEmailContext, claimWindowSlot } from "./auth-email-cooldown";

// Documented in docs/platform/rate-limits.md: update it with any change here.
// The wrong codes one address may get in a 30-day window, across all its
// codes. Five tries a code and five codes an hour would otherwise let anyone
// whose provider vouches for the address keep guessing for months. Past this,
// no code goes to the address until the window ends; its owner signs in the
// usual way.
const WRONG_CODES_PER_ADDRESS = { max: 20, windowSeconds: 30 * 24 * 60 * 60 };

export function wrongCodeIdentifier(email: string) {
  const key = createHash("sha256").update(email.toLowerCase()).digest("hex");
  return `sign-in-code-wrong:${key}`;
}

/**
 * Counts a wrong code against its address: one row each, which only one
 * request can insert, so codes checked at once all count. "locked" once the
 * address has none left; "failed" when the count can't be kept.
 */
export async function recordWrongCode(
  context: AuthEmailContext,
  email: string,
): Promise<"counted" | "locked" | "failed"> {
  try {
    const slot = await claimWindowSlot(
      context,
      wrongCodeIdentifier(email),
      WRONG_CODES_PER_ADDRESS,
    );
    return slot === null ? "locked" : "counted";
  } catch (error) {
    logFailure("Failed to count a wrong sign-in code", error);
    return "failed";
  }
}

/** Whether the address has had its wrong codes for now, or can't be checked. */
export async function isLockedOut(context: AuthEmailContext, email: string) {
  try {
    const counted = await context.adapter.findMany({
      model: "verification",
      where: [
        { field: "identifier", value: wrongCodeIdentifier(email) },
        { field: "expiresAt", value: new Date(), operator: "gt" },
      ],
      limit: WRONG_CODES_PER_ADDRESS.max,
    });
    return counted.length >= WRONG_CODES_PER_ADDRESS.max;
  } catch (error) {
    logFailure("Failed to check an address's wrong sign-in codes", error);
    return true;
  }
}

function logFailure(message: string, error: unknown) {
  console.error(message, {
    error: error instanceof Error ? error.message : String(error),
  });
}
