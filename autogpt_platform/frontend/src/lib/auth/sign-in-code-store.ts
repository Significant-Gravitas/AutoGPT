import type { GenericEndpointContext } from "@better-auth/core";
import * as z from "zod";
import {
  type AuthEmailContext,
  claimOneOf,
  createVerification,
} from "./auth-email-cooldown";
import {
  SIGN_IN_CODE_RESEND_SECONDS,
  SIGN_IN_CODE_TTL_SECONDS,
} from "./sign-in-code";
import { CODE_PROVIDER_IDS } from "./mailbox-provider";
import { pendingCookieName } from "./sign-in-code-cookie";

// A sign-in waiting on its code: a `sign-in-code:<id>` row in Better Auth's
// verification table until the code expires, and the id in a signed cookie
// in the browser that started it.

// Documented in docs/platform/rate-limits.md: update it with any change here.
export const MAX_CODE_ATTEMPTS = 5;

const RECORD_PREFIX = "sign-in-code:";
const TRY_PREFIX = "sign-in-code-try:";

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
  // The row's value as this request read it: the version of the code it saw,
  // which its later writes require so they can't undo another request's.
  value: string;
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
    pendingCookieName(ctx),
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
    value: row.value,
  };
}

/**
 * Puts a new code in place of the one this request read, if it is still
 * there and still good: of two resends at once, one wins. The new version, or
 * null when another request changed the row first or the sign-in ran out.
 */
export async function replacePending(
  context: AuthEmailContext,
  pending: Pending,
  record: PendingRecord,
  expiresAt: Date,
) {
  const value = JSON.stringify(record);
  const updated = await context.adapter.updateMany({
    model: "verification",
    where: [
      ...thisVersion(pending.rowID, pending.value),
      // Not a sign-in that ran out since it was read.
      { field: "expiresAt", value: new Date(), operator: "gt" },
    ],
    update: { value, expiresAt, updatedAt: new Date() },
  });
  return Number(updated) > 0 ? value : null;
}

/** Undoes `replacePending` if nothing has changed the row since. */
export async function restorePending(
  context: AuthEmailContext,
  pending: Pending,
  replacedWith: string,
) {
  await context.adapter.updateMany({
    model: "verification",
    where: thisVersion(pending.rowID, replacedWith),
    update: {
      value: pending.value,
      expiresAt: pending.expiresAt,
      updatedAt: new Date(),
    },
  });
}

/**
 * Removes the version of the waiting sign-in this request checked, while its
 * code is still good. Whether this call removed it: of two requests with the
 * right code one finishes, and a check of a code a resend has replaced
 * removes nothing.
 */
export async function consumePending(
  context: AuthEmailContext,
  pending: Pending,
) {
  const deleted = await context.adapter.deleteMany({
    model: "verification",
    where: [
      ...thisVersion(pending.rowID, pending.value),
      { field: "expiresAt", value: new Date(), operator: "gt" },
    ],
  });
  return Number(deleted) > 0;
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

function thisVersion(rowID: string, value: string) {
  return [
    { field: "id", value: rowID },
    { field: "value", value },
  ];
}

function parseJSON(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}
