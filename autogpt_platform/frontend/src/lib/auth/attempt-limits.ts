import { createHash } from "node:crypto";
import { defineRequestState } from "@better-auth/core/context";
import type { BetterAuthOptions } from "better-auth";
import { APIError, getIP, isAPIError } from "better-auth/api";
import {
  type AuthEmailContext,
  claimWindowSlot,
  releaseWindowSlot,
  type WindowLimit,
} from "./auth-email-cooldown";
import { getTrustedClientIP } from "./client-ip";

interface PathLimits {
  name: string;
  perIP: WindowLimit;
  perAddress: WindowLimit;
  // A successful attempt gives its address slot back, so only failures count.
  refundOnSuccess?: boolean;
}

// Per IP, the rules Better Auth's own limiter applies to these routes over
// HTTP. Per address, whether or not it has an account, so a refusal can't tell
// anyone which addresses do.
export const ATTEMPT_LIMITS = new Map<string, PathLimits>([
  [
    "/sign-in/email",
    {
      name: "sign-in",
      perIP: { max: 3, windowSeconds: 10 },
      perAddress: { max: 5, windowSeconds: 10 * 60 },
      refundOnSuccess: true,
    },
  ],
  [
    "/request-password-reset",
    {
      name: "password-reset",
      perIP: { max: 3, windowSeconds: 60 },
      perAddress: { max: 1, windowSeconds: 10 * 60 },
    },
  ],
]);

// The address slot this call claimed, for refundSuccessfulAttempt.
const claimedAddressSlot = defineRequestState<string | null>(() => null);

interface Args {
  path: string | undefined;
  body: unknown;
  headers: Headers | undefined;
  context: AuthEmailContext & { options: BetterAuthOptions };
}

/**
 * A Better Auth before-hook, so it counts the login and reset-password pages'
 * server actions (auth.api) and the HTTP routes alike. The counts live in the
 * verification table, which every server shares. With no client IP we trust
 * (client-ip.ts) only the address is limited, never one bucket for everyone.
 * If a count can't be read, the attempt goes through, as with the email caps.
 */
export async function limitAuthAttempts({
  path,
  body,
  headers,
  context,
}: Args) {
  const limits = path ? ATTEMPT_LIMITS.get(path) : undefined;
  if (!limits) return;
  const ip = headers
    ? getTrustedClientIP(getIP(headers, context.options))
    : null;
  if (ip) await claimOrRefuse(context, `${limits.name}-ip:${ip}`, limits.perIP);
  const address = addressIn(body);
  if (address) {
    // Hashed: this runs before Better Auth validates the body, so the address
    // can be any length.
    const addressKey = createHash("sha256").update(address).digest("hex");
    const slot = await claimOrRefuse(
      context,
      `${limits.name}-address:${addressKey}`,
      limits.perAddress,
    );
    if (slot && limits.refundOnSuccess) await claimedAddressSlot.set(slot);
  }
}

/** A Better Auth after-hook: a successful sign-in frees the slot it claimed. */
export async function refundSuccessfulAttempt({
  path,
  returned,
  context,
}: {
  path: string | undefined;
  returned: unknown;
  context: AuthEmailContext;
}) {
  if (!path || !ATTEMPT_LIMITS.get(path)?.refundOnSuccess) return;
  if (isAPIError(returned)) return;
  const slot = await claimedAddressSlot.get();
  if (!slot) return;
  await claimedAddressSlot.set(null);
  await releaseWindowSlot(context, slot).catch((error: unknown) => {
    console.error("Failed to release an auth attempt slot", {
      error: error instanceof Error ? error.message : String(error),
    });
  });
}

// The claimed slot's id, or null when the count couldn't be read.
async function claimOrRefuse(
  context: AuthEmailContext,
  identifier: string,
  limit: WindowLimit,
) {
  let slot: string | null;
  try {
    slot = await claimWindowSlot(context, identifier, limit);
  } catch (error) {
    console.error("Failed to check an auth attempt limit", {
      error: error instanceof Error ? error.message : String(error),
    });
    return null;
  }
  if (slot) return slot;
  const retryAfter = secondsLeftInWindow(limit.windowSeconds);
  throw new APIError(
    "TOO_MANY_REQUESTS",
    {
      message: `Too many attempts. Please try again in ${describeWait(retryAfter)}.`,
    },
    { "X-Retry-After": String(retryAfter) },
  );
}

function addressIn(body: unknown) {
  if (typeof body !== "object" || body === null || !("email" in body)) {
    return null;
  }
  return typeof body.email === "string"
    ? body.email.trim().toLowerCase() || null
    : null;
}

function secondsLeftInWindow(windowSeconds: number) {
  const elapsed = (Date.now() / 1000) % windowSeconds;
  return Math.max(1, Math.ceil(windowSeconds - elapsed));
}

function describeWait(seconds: number) {
  if (seconds < 60) return seconds === 1 ? "1 second" : `${seconds} seconds`;
  const minutes = Math.ceil(seconds / 60);
  return minutes === 1 ? "1 minute" : `${minutes} minutes`;
}
