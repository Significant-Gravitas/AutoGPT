import { createHash } from "node:crypto";

// At most one email of a kind per address per window, however often whatever
// sends it is repeated.
const AUTH_EMAIL_COOLDOWN_SECONDS = 10 * 60;
// And at most this many sign-ups and resends per IP per window, so fresh
// addresses (or +aliases of one mailbox) can't each be mailed.
export const AUTH_EMAILS_PER_IP = 5;

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
      id: slotID(identifier),
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

function slotID(identifier: string, index?: number) {
  const window = currentWindow();
  const key =
    index === undefined
      ? `${identifier}#${window}`
      : `${identifier}#${window}#${index}`;
  return createHash("sha256").update(key).digest("hex");
}

function currentWindow() {
  return Math.floor(Date.now() / (AUTH_EMAIL_COOLDOWN_SECONDS * 1000));
}

// One of AUTH_EMAILS_PER_IP fixed rows per IP and window; each id can only be
// inserted once, so a burst from one IP can't take more than that many. A
// failed insert whose row now exists lost the race and tries the next slot.
export async function claimIPEmailSlot(context: AuthEmailContext, ip: string) {
  const identifier = `auth-email-ip:${ip}`;
  const windowEndsAt = (currentWindow() + 1) * AUTH_EMAIL_COOLDOWN_SECONDS;
  for (let index = 0; index < AUTH_EMAILS_PER_IP; index++) {
    const id = slotID(identifier, index);
    if (await rowExists(context, id)) continue;
    try {
      await createVerification(context, {
        id,
        identifier,
        value: "sent",
        expiresInSeconds: windowEndsAt - Date.now() / 1000,
      });
      return true;
    } catch (error) {
      if (!(await rowExists(context, id))) throw error;
    }
  }
  return false;
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
