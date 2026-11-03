import { createHash } from "node:crypto";

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
  value: string | Date;
  operator?: "eq" | "gt" | "starts_with" | "ends_with";
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
  for (let index = 0; index < max; index++) {
    const id = slotID(identifier, window, index);
    if (await rowExists(context, id)) continue;
    try {
      await createVerification(context, {
        id,
        identifier,
        value: "sent",
        expiresInSeconds: windowEndsAt - Date.now() / 1000,
      });
      return id;
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
  const rows = await context.adapter.findMany({
    model: "verification",
    where: [{ field: "id", value: id }],
    limit: 1,
  });
  return rows.length > 0;
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
