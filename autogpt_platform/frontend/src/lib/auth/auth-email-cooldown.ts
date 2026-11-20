import { createHash } from "node:crypto";

// Documented in docs/platform/rate-limits.md: update it with any change here.
// At most one email of a kind per address per window, however often whatever
// sends it is repeated.
const AUTH_EMAIL_COOLDOWN_SECONDS = 10 * 60;
// And at most this many sign-ups and resends per IP per window, so fresh
// addresses (or +aliases of one mailbox) can't each be mailed.
export const AUTH_EMAILS_PER_IP = 5;

export interface WindowLimit {
  max: number;
  windowSeconds: number;
}

interface Where {
  field: string;
  value: string | string[] | Date;
  operator?: "eq" | "gt" | "in" | "starts_with" | "ends_with";
}

/**
 * The parts of Better Auth's context these emails use. They run alongside the
 * response (see background-tasks.ts), after a sign-up's database transaction
 * has committed, and Better Auth's internal adapter (and any auth.api call)
 * would still go through that transaction. So they use the base adapter.
 */
export interface AuthEmailContext {
  baseURL: string;
  adapter: {
    findMany: (args: {
      model: string;
      where: Where[];
      limit: number;
    }) => Promise<unknown[]>;
    create: (args: {
      model: string;
      data: Record<string, unknown>;
      forceAllowId?: boolean;
    }) => Promise<unknown>;
    updateMany: (args: {
      model: string;
      where: Where[];
      update: Record<string, unknown>;
    }) => Promise<unknown>;
    deleteMany: (args: { model: string; where: Where[] }) => Promise<unknown>;
  };
  password: { hash: (password: string) => Promise<string> };
}

// Kept in Better Auth's verification table, which every server shares and
// which Better Auth clears of expired rows itself. The row's id is fixed per
// address and window, so when several sends pass the check below at the same
// moment, the primary key lets one insert win and the rest fail.
export async function claimEmailSlot(
  context: AuthEmailContext,
  kind: "repeat-sign-up" | "verify-email",
  email: string,
) {
  const identifier = `${kind}:${email.toLowerCase()}`;
  if (await hasLiveSlot(context, identifier)) return false;
  try {
    await createVerification(context, {
      id: slotID(identifier, currentWindow(AUTH_EMAIL_COOLDOWN_SECONDS)),
      identifier,
      value: "sent",
      expiresInSeconds: AUTH_EMAIL_COOLDOWN_SECONDS,
    });
  } catch (error) {
    // Lost the race to a concurrent send, or a real failure.
    if (await hasLiveSlot(context, identifier)) return false;
    throw error;
  }
  return true;
}

async function hasLiveSlot(context: AuthEmailContext, identifier: string) {
  const live = await context.adapter.findMany({
    model: "verification",
    where: [
      { field: "identifier", value: identifier },
      { field: "expiresAt", value: new Date(), operator: "gt" },
    ],
    limit: 1,
  });
  return live.length > 0;
}

function slotID(identifier: string, window: number, index?: number) {
  const key =
    index === undefined
      ? `${identifier}#${window}`
      : `${identifier}#${window}#${index}`;
  return createHash("sha256").update(key).digest("hex");
}

function currentWindow(windowSeconds: number) {
  return Math.floor(Date.now() / (windowSeconds * 1000));
}

export async function claimIPEmailSlot(context: AuthEmailContext, ip: string) {
  const slot = await claimWindowSlot(context, `auth-email-ip:${ip}`, {
    max: AUTH_EMAILS_PER_IP,
    windowSeconds: AUTH_EMAIL_COOLDOWN_SECONDS,
  });
  return slot !== null;
}

// One of `max` fixed rows per identifier and window; each id can only be
// inserted once, so a burst can't take more than that many. A failed insert
// whose row now exists lost the race and tries the next slot. Returns the
// claimed row's id, or null when every slot is taken.
export async function claimWindowSlot(
  context: AuthEmailContext,
  identifier: string,
  { max, windowSeconds }: WindowLimit,
) {
  const window = currentWindow(windowSeconds);
  const windowEndsAt = (window + 1) * windowSeconds;
  const slot = await claimSlot(
    context,
    identifier,
    max,
    window,
    windowEndsAt - Date.now() / 1000,
  );
  return slot?.id ?? null;
}

// The same race-safe slots for an identifier that is never reused (one
// emailed code's tries), kept until it expires instead of per clock window.
// Returns the claimed slot's index, from 0, or null when all `max` are taken.
export async function claimOneOf(
  context: AuthEmailContext,
  identifier: string,
  max: number,
  expiresInSeconds: number,
) {
  const slot = await claimSlot(context, identifier, max, 0, expiresInSeconds);
  return slot?.index ?? null;
}

async function claimSlot(
  context: AuthEmailContext,
  identifier: string,
  max: number,
  window: number,
  expiresInSeconds: number,
) {
  const ids = Array.from({ length: max }, (_, index) =>
    slotID(identifier, window, index),
  );
  const taken = await existingIDs(context, ids);
  for (const [index, id] of ids.entries()) {
    if (taken.has(id)) continue;
    try {
      await createVerification(context, {
        id,
        identifier,
        value: "sent",
        expiresInSeconds,
      });
      return { id, index };
    } catch (error) {
      if (!(await rowExists(context, id))) throw error;
    }
  }
  return null;
}

export async function releaseWindowSlot(context: AuthEmailContext, id: string) {
  await context.adapter.deleteMany({
    model: "verification",
    where: [{ field: "id", value: id }],
  });
}

async function rowExists(context: AuthEmailContext, id: string) {
  return (await existingIDs(context, [id])).size > 0;
}

// Which of these rows are there, in one read.
async function existingIDs(context: AuthEmailContext, ids: string[]) {
  const rows = (await context.adapter.findMany({
    model: "verification",
    where: [{ field: "id", value: ids, operator: "in" }],
    limit: ids.length,
  })) as Array<{ id: string }>;
  return new Set(rows.map((row) => row.id));
}

export function createVerification(
  context: AuthEmailContext,
  args: {
    id?: string;
    identifier: string;
    value: string;
    expiresInSeconds: number;
  },
) {
  const now = new Date();
  return context.adapter.create({
    model: "verification",
    forceAllowId: args.id !== undefined,
    data: {
      ...(args.id === undefined ? {} : { id: args.id }),
      identifier: args.identifier,
      value: args.value,
      expiresAt: new Date(now.getTime() + args.expiresInSeconds * 1000),
      createdAt: now,
      updatedAt: now,
    },
  });
}
