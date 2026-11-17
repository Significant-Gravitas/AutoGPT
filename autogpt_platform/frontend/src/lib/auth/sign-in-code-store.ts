import type { GenericEndpointContext } from "@better-auth/core";
import { APIError } from "better-auth/api";
import { expireCookie } from "better-auth/cookies";
import * as z from "zod";
import {
  type AuthEmailContext,
  claimOneOf,
  createVerification,
} from "./auth-email-cooldown";
import {
  SIGN_IN_CODE_EXPIRED,
  SIGN_IN_CODE_RESEND_SECONDS,
  SIGN_IN_CODE_TTL_SECONDS,
} from "./sign-in-code";

// A sign-in waiting on its code: a `sign-in-code:<id>` row in Better Auth's
// verification table until the code expires, and the id in a signed cookie
// in the browser that started it.

// Documented in docs/platform/rate-limits.md: update it with any change here.
export const MAX_CODE_ATTEMPTS = 5;

// The providers whose sign-ins the code covers. Better Auth trusts exactly
// these to link (auth.ts), so the hook that asks for the code stands in for
// its verified-address check: keep the two one list.
export const CODE_PROVIDER_IDS = [
  "google",
  "microsoft",
  "github",
  "discord",
] as const;

const RECORD_PREFIX = "sign-in-code:";
const TRY_PREFIX = "sign-in-code-try:";
const COOKIE_NAME = "sign_in_code";

const pendingSchema = z.object({
  providerId: z.enum(CODE_PROVIDER_IDS),
  accountId: z.string().min(1),
  email: z.string().min(1),
  name: z.string(),
  image: z.string().nullable(),
  codeHash: z.string(),
  sentAt: z.number(),
});
export type PendingRecord = z.infer<typeof pendingSchema>;
export type PendingSignIn = Omit<PendingRecord, "codeHash" | "sentAt">;

export interface Pending {
  id: string;
  rowID: string;
  record: PendingRecord;
  expiresAt: Date;
}

export async function storePending(
  context: AuthEmailContext,
  id: string,
  record: PendingRecord,
) {
  await createVerification(context, {
    identifier: RECORD_PREFIX + id,
    value: JSON.stringify(record),
    expiresInSeconds: SIGN_IN_CODE_TTL_SECONDS,
  });
}

/** The sign-in this browser's cookie names, while its code is good. */
export async function findPending(
  ctx: GenericEndpointContext,
): Promise<Pending | null> {
  const id = await ctx.getSignedCookie(
    pendingCookie(ctx).name,
    ctx.context.secret,
  );
  if (typeof id !== "string" || !id) return null;
  const [row] = (await ctx.context.adapter.findMany({
    model: "verification",
    where: [
      { field: "identifier", value: RECORD_PREFIX + id },
      { field: "expiresAt", value: new Date(), operator: "gt" },
    ],
    limit: 1,
  })) as Array<{ id: string; value: string; expiresAt: Date | string }>;
  if (!row) return null;
  const parsed = pendingSchema.safeParse(parseJSON(row.value));
  if (!parsed.success) return null;
  return {
    id,
    rowID: row.id,
    record: parsed.data,
    expiresAt: new Date(row.expiresAt),
  };
}

export async function savePending(
  context: AuthEmailContext,
  rowID: string,
  record: PendingRecord,
  expiresAt?: Date,
) {
  await context.adapter.updateMany({
    model: "verification",
    where: [{ field: "id", value: rowID }],
    update: {
      value: JSON.stringify(record),
      updatedAt: new Date(),
      ...(expiresAt ? { expiresAt } : {}),
    },
  });
}

// Whether this call removed the row, so a second request with the same code
// can't finish the sign-in again.
export async function deletePending(context: AuthEmailContext, rowID: string) {
  const deleted = await context.adapter.deleteMany({
    model: "verification",
    where: [{ field: "id", value: rowID }],
  });
  return typeof deleted !== "number" || deleted > 0;
}

export async function forgetPending(context: AuthEmailContext, id: string) {
  await context.adapter.deleteMany({
    model: "verification",
    where: [{ field: "identifier", value: RECORD_PREFIX + id }],
  });
}

/**
 * Takes one of the current code's tries, before its check compares anything.
 * A try is a row only one request can insert, so checks sent at once can't
 * share one; a new code gets its own. The try's index, or null when all are
 * gone.
 */
export function claimTry(context: AuthEmailContext, pending: Pending) {
  const left = (pending.expiresAt.getTime() - Date.now()) / 1000;
  return claimOneOf(
    context,
    `${TRY_PREFIX}${pending.id}:${pending.record.sentAt}`,
    MAX_CODE_ATTEMPTS,
    Math.max(1, left),
  );
}

export function secondsUntilResend(record: PendingRecord) {
  const due = record.sentAt + SIGN_IN_CODE_RESEND_SECONDS * 1000;
  return Math.max(0, Math.ceil((due - Date.now()) / 1000));
}

export async function setPendingCookie(
  ctx: GenericEndpointContext,
  id: string,
) {
  const cookie = pendingCookie(ctx);
  await ctx.setSignedCookie(
    cookie.name,
    id,
    ctx.context.secret,
    cookie.attributes,
  );
}

export function forgetPendingCookie(ctx: GenericEndpointContext) {
  expireCookie(ctx, pendingCookie(ctx));
}

/** The refusal when nothing waits on a code here; it drops the cookie too. */
export function expiredError(ctx: GenericEndpointContext) {
  forgetPendingCookie(ctx);
  return new APIError("BAD_REQUEST", {
    code: SIGN_IN_CODE_EXPIRED,
    message: "This sign-in has expired. Start again.",
  });
}

function pendingCookie(ctx: GenericEndpointContext) {
  return ctx.context.createAuthCookie(COOKIE_NAME, {
    maxAge: SIGN_IN_CODE_TTL_SECONDS,
  });
}

function parseJSON(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}
