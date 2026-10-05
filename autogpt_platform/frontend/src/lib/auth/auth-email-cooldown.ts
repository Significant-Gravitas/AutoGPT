import { createHash } from "node:crypto";
import { sendAuthEmail } from "./email";

// At most one email of a kind per address per window, however often whatever
// sends it is repeated.
const AUTH_EMAIL_COOLDOWN_SECONDS = 10 * 60;

interface Where {
  field: string;
  value: string | Date;
  operator?: "eq" | "gt";
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
  };
  password: { hash: (password: string) => Promise<string> };
}

interface VerificationLinkArgs {
  user: { email: string };
  url: string;
  request?: Request;
  getAuthContext: () => Promise<AuthEmailContext>;
}

/**
 * Better Auth's sendVerificationEmail. Sign-in sends one every time an
 * unverified account signs in, and the login page calls auth.api directly,
 * which Better Auth's rate limiter never sees, so whoever set the password
 * could have us mail the address without limit. Sign-in and sign-up therefore
 * share one email per address per window.
 *
 * The resend button's own route is left alone: Better Auth rate-limits it per
 * IP, the button waits a minute between sends, and its answer has to say
 * whether the email went. If the cooldown can't be checked, the email still
 * goes: a verification link matters more than the cap.
 */
export async function sendVerificationLink({
  user,
  url,
  request,
  getAuthContext,
}: VerificationLinkArgs) {
  if (!isResendRequest(request)) {
    const claimed = await getAuthContext()
      .then((context) => claimEmailSlot(context, "verify-email", user.email))
      .catch((error: unknown) => {
        console.error("Failed to check the verification email cooldown", {
          error: error instanceof Error ? error.message : String(error),
        });
        return true;
      });
    if (!claimed) return;
  }
  await sendAuthEmail({ to: user.email, type: "verify_email", url });
}

function isResendRequest(request: Request | undefined) {
  if (!request) return false;
  return new URL(request.url).pathname.endsWith("/send-verification-email");
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

function slotID(identifier: string) {
  const window = Math.floor(Date.now() / (AUTH_EMAIL_COOLDOWN_SECONDS * 1000));
  return createHash("sha256").update(`${identifier}#${window}`).digest("hex");
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
