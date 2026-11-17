import {
  createHash,
  createHmac,
  randomBytes,
  randomInt,
  timingSafeEqual,
} from "node:crypto";
import type { GenericEndpointContext } from "@better-auth/core";
import type {
  BetterAuthPlugin,
  ValidateUserInfoResult,
  ValidateUserInfoSource,
} from "better-auth";
import { APIError, createAuthEndpoint } from "better-auth/api";
import { expireCookie, setSessionCookie } from "better-auth/cookies";
import { handleOAuthUserInfo } from "better-auth/oauth2";
import * as z from "zod";
import { limitAttemptsPerIP } from "./attempt-limits";
import {
  type AuthEmailContext,
  claimWindowSlot,
  createVerification,
} from "./auth-email-cooldown";
import { sendAuthEmail, type SignInEmailProvider } from "./email";
import { providerRunsMailbox } from "./mailbox-provider";
import {
  type PendingSignInCode,
  SIGN_IN_CODE_EXPIRED,
  SIGN_IN_CODE_INVALID,
  SIGN_IN_CODE_LENGTH,
  SIGN_IN_CODE_LIMIT,
  SIGN_IN_CODE_PATHS,
  SIGN_IN_CODE_RESEND_SECONDS,
  SIGN_IN_CODE_SENT,
  SIGN_IN_CODE_TOO_SOON,
  SIGN_IN_CODE_TTL_SECONDS,
  SIGN_IN_CODE_UNSENT,
} from "./sign-in-code";
import {
  markConfirmedByCode,
  providerConfirmedByCode,
} from "./sign-in-provider";

/**
 * A provider sign-in that can't vouch for its address finishes with a code
 * emailed there, typed into the page that started it:
 *
 * - Joining an existing account, unless the provider runs that mailbox
 *   (mailbox-provider.ts). Better Auth would otherwise link any provider that
 *   says it verified the address, and a provider that vouched for addresses
 *   it never verified would open every account that uses them.
 * - Creating an account for an address the provider didn't verify, which
 *   provider-verified-email.ts would otherwise refuse.
 *
 * The code can't be a link: the mailbox's owner could then approve someone
 * else's sign-in with a click. It is checked against the browser that started
 * the sign-in (a signed cookie), and finishing it runs Better Auth's own
 * sign-in with the provider account the callback saw, so every sign-up check
 * and session hook still applies.
 *
 * Apple keeps Better Auth's rule: Apple verifies every address it hands out,
 * and its native sign-in has no page to type a code into.
 */

// Documented in docs/platform/rate-limits.md: update it with any change here.
const MAX_CODE_ATTEMPTS = 5;
const CODE_EMAILS_PER_ADDRESS = { max: 5, windowSeconds: 60 * 60 };
const CODE_CHECKS_PER_IP = { max: 10, windowSeconds: 60 };

const RECORD_PREFIX = "sign-in-code:";
const COOKIE_NAME = "sign_in_code";

const CODE_PROVIDERS: ReadonlySet<string> = new Set([
  "google",
  "microsoft",
  "github",
  "discord",
]);
// The providers a sign-in method notice can name.
const NOTICE_PROVIDERS: ReadonlySet<string> = new Set([
  ...CODE_PROVIDERS,
  "apple",
]);

const pendingSchema = z.object({
  providerId: z.enum(["google", "microsoft", "github", "discord"]),
  accountId: z.string().min(1),
  email: z.string().min(1),
  name: z.string(),
  image: z.string().nullable(),
  codeHash: z.string(),
  attempts: z.number().int().min(0),
  sentAt: z.number(),
});
type PendingRecord = z.infer<typeof pendingSchema>;
type PendingSignIn = Omit<PendingRecord, "codeHash" | "attempts" | "sentAt">;

interface Pending {
  id: string;
  rowId: string;
  record: PendingRecord;
  expiresAt: Date;
}

/**
 * Better Auth's `user.validateUserInfo`. It runs before a provider sign-in
 * joins an existing account or creates one; returning an error stops it and
 * sends the browser to the sign-in's error URL with that code (`/login`, which
 * then asks for the code).
 */
export async function requireSignInCode(
  data: { user: Record<string, unknown>; source: ValidateUserInfoSource },
  ctx: GenericEndpointContext,
): Promise<ValidateUserInfoResult | undefined> {
  const { user, source } = data;
  if (source.method !== "oauth" || !source.oauth) return;
  const { providerId, profile } = source.oauth;
  if (!isCodeProvider(providerId)) return;
  // The code's own endpoint, finishing the sign-in it was emailed for.
  if (providerConfirmedByCode(ctx) === providerId) return;
  const email = typeof user.email === "string" ? user.email.toLowerCase() : "";
  if (!email) return;
  const emailVerified = user.emailVerified === true;
  if (source.action === "link-account") {
    if (providerRunsMailbox(providerId, { email, emailVerified, profile })) {
      return;
    }
  } else if (source.action !== "create-user" || emailVerified) {
    return;
  }
  const accountId = await accountIdOf(ctx, providerId, profile);
  if (!accountId) return { error: SIGN_IN_CODE_UNSENT };
  return startSignInCode(ctx, {
    providerId,
    accountId,
    email,
    name: typeof user.name === "string" ? user.name : "",
    image: typeof user.image === "string" ? user.image : null,
  });
}

/** The endpoints /login uses while a sign-in waits on its code. */
export function signInCode() {
  return {
    id: "sign-in-code",
    endpoints: {
      signInCodeStatus: createAuthEndpoint(
        SIGN_IN_CODE_PATHS.status,
        { method: "GET" },
        async (ctx) => {
          const pending = await findPending(ctx);
          if (!pending) return ctx.json({ pending: null });
          const { record } = pending;
          const answer: PendingSignInCode = {
            provider: record.providerId,
            email: record.email,
            resendInSeconds: secondsUntilResend(record),
          };
          return ctx.json({ pending: answer });
        },
      ),
      signInCodeVerify: createAuthEndpoint(
        SIGN_IN_CODE_PATHS.verify,
        { method: "POST", body: z.object({ code: z.string().max(64) }) },
        async (ctx) => {
          await limitAttemptsPerIP({
            name: "sign-in-code",
            limit: CODE_CHECKS_PER_IP,
            headers: ctx.request?.headers ?? ctx.headers,
            context: ctx.context,
          });
          const pending = await findPending(ctx);
          if (!pending) throw expired(ctx);
          const { id, rowId, record } = pending;
          const code = ctx.body.code.replace(/\s+/g, "");
          if (!codeMatches(ctx.context.secret, id, code, record.codeHash)) {
            const attempts = record.attempts + 1;
            if (attempts >= MAX_CODE_ATTEMPTS) {
              await deletePending(ctx.context, rowId);
              throw expired(ctx);
            }
            await savePending(ctx.context, rowId, { ...record, attempts });
            throw new APIError("BAD_REQUEST", {
              code: SIGN_IN_CODE_INVALID,
              message: "That code isn't right.",
            });
          }
          // One use: of two requests with the right code, one finishes.
          if (!(await deletePending(ctx.context, rowId))) throw expired(ctx);
          expireCookie(ctx, pendingCookie(ctx));
          return finishSignIn(ctx, record);
        },
      ),
      signInCodeResend: createAuthEndpoint(
        SIGN_IN_CODE_PATHS.resend,
        { method: "POST" },
        async (ctx) => {
          const pending = await findPending(ctx);
          if (!pending) throw expired(ctx);
          const { id, rowId, record, expiresAt } = pending;
          const wait = secondsUntilResend(record);
          if (wait > 0) {
            throw new APIError(
              "TOO_MANY_REQUESTS",
              {
                code: SIGN_IN_CODE_TOO_SOON,
                message: `Wait ${wait} seconds before asking for a new code.`,
              },
              { "X-Retry-After": String(wait) },
            );
          }
          if (!(await claimCodeEmail(ctx.context, record.email))) {
            throw new APIError("TOO_MANY_REQUESTS", {
              code: SIGN_IN_CODE_LIMIT,
              message: "Too many codes for this address.",
            });
          }
          const code = newCode();
          const sentAt = Date.now();
          await savePending(
            ctx.context,
            rowId,
            {
              ...record,
              codeHash: hashCode(ctx.context.secret, id, code),
              attempts: 0,
              sentAt,
            },
            new Date(sentAt + SIGN_IN_CODE_TTL_SECONDS * 1000),
          );
          if (!(await sendCode(record, code))) {
            // The old code still works, as long as it did, and the email can
            // be asked for again.
            await savePending(ctx.context, rowId, record, expiresAt);
            throw new APIError("SERVICE_UNAVAILABLE", {
              code: SIGN_IN_CODE_UNSENT,
              message: "The code couldn't be sent.",
            });
          }
          await setPendingCookie(ctx, id);
          return ctx.json({ resendInSeconds: SIGN_IN_CODE_RESEND_SECONDS });
        },
      ),
    },
  } satisfies BetterAuthPlugin;
}

/**
 * Emails the account's owner when a sign-in provider joins an account it
 * didn't create. Called after the account row is written; a provider's first
 * row on a new account is its sign-up, which needs no notice.
 */
export async function alertNewSignInMethod(
  context: AuthEmailContext,
  account: { userId: string; providerId: string },
) {
  if (!isNoticeProvider(account.providerId)) return;
  const accounts = await context.adapter.findMany({
    model: "account",
    where: [{ field: "userId", value: account.userId }],
    limit: 2,
  });
  if (accounts.length < 2) return;
  const [user] = (await context.adapter.findMany({
    model: "user",
    where: [{ field: "id", value: account.userId }],
    limit: 1,
  })) as Array<{ email?: string }>;
  if (!user?.email) return;
  await sendAuthEmail({
    to: user.email,
    type: "sign_in_method_added",
    provider: account.providerId,
  });
}

async function startSignInCode(
  ctx: GenericEndpointContext,
  pending: PendingSignIn,
): Promise<ValidateUserInfoResult> {
  if (!(await claimCodeEmail(ctx.context, pending.email))) {
    return { error: SIGN_IN_CODE_LIMIT };
  }
  const id = randomBytes(32).toString("base64url");
  const code = newCode();
  const record: PendingRecord = {
    ...pending,
    codeHash: hashCode(ctx.context.secret, id, code),
    attempts: 0,
    sentAt: Date.now(),
  };
  await createVerification(ctx.context, {
    identifier: RECORD_PREFIX + id,
    value: JSON.stringify(record),
    expiresInSeconds: SIGN_IN_CODE_TTL_SECONDS,
  });
  if (!(await sendCode(pending, code))) {
    await ctx.context.adapter.deleteMany({
      model: "verification",
      where: [{ field: "identifier", value: RECORD_PREFIX + id }],
    });
    return { error: SIGN_IN_CODE_UNSENT };
  }
  await setPendingCookie(ctx, id);
  return { error: SIGN_IN_CODE_SENT };
}

// Better Auth's own sign-in for the provider account the callback saw, with
// the address now proven. It links that account to the address's account, or
// creates one, through every database hook, then starts the session.
async function finishSignIn(
  ctx: GenericEndpointContext,
  record: PendingRecord,
) {
  markConfirmedByCode(ctx, record.providerId);
  const provider = await findProvider(ctx, record.providerId);
  if (!provider) throw expired(ctx);
  const result = await handleOAuthUserInfo(ctx, {
    userInfo: {
      id: record.accountId,
      email: record.email,
      emailVerified: true,
      name: record.name,
      image: record.image,
    },
    account: { providerId: record.providerId, accountId: record.accountId },
    disableSignUp: provider.options?.disableSignUp === true,
    requireExactAccountBinding: true,
    source: { method: "oauth", oauth: { providerId: record.providerId } },
  });
  if (!result.data) {
    // Better Auth's sign-in errors, as its callback sends them to /login.
    throw new APIError("FORBIDDEN", {
      code: result.error.split(" ").join("_"),
      message: "",
    });
  }
  await setSessionCookie(ctx, result.data);
  return ctx.json({
    provider: record.providerId,
    newUser: result.isRegister === true,
  });
}

async function findPending(
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
    rowId: row.id,
    record: parsed.data,
    expiresAt: new Date(row.expiresAt),
  };
}

async function savePending(
  context: AuthEmailContext,
  rowId: string,
  record: PendingRecord,
  expiresAt?: Date,
) {
  await context.adapter.updateMany({
    model: "verification",
    where: [{ field: "id", value: rowId }],
    update: {
      value: JSON.stringify(record),
      updatedAt: new Date(),
      ...(expiresAt ? { expiresAt } : {}),
    },
  });
}

// Whether this call removed the row, so a second request with the same code
// can't finish the sign-in again.
async function deletePending(context: AuthEmailContext, rowId: string) {
  const deleted = await context.adapter.deleteMany({
    model: "verification",
    where: [{ field: "id", value: rowId }],
  });
  return typeof deleted !== "number" || deleted > 0;
}

function pendingCookie(ctx: GenericEndpointContext) {
  return ctx.context.createAuthCookie(COOKIE_NAME, {
    maxAge: SIGN_IN_CODE_TTL_SECONDS,
  });
}

async function setPendingCookie(ctx: GenericEndpointContext, id: string) {
  const cookie = pendingCookie(ctx);
  await ctx.setSignedCookie(
    cookie.name,
    id,
    ctx.context.secret,
    cookie.attributes,
  );
}

function expired(ctx: GenericEndpointContext) {
  expireCookie(ctx, pendingCookie(ctx));
  return new APIError("BAD_REQUEST", {
    code: SIGN_IN_CODE_EXPIRED,
    message: "This sign-in has expired. Start again.",
  });
}

// At most CODE_EMAILS_PER_ADDRESS codes an hour, from sign-ins and resends
// together. If the count can't be read the code goes out, as with the other
// auth emails.
async function claimCodeEmail(context: AuthEmailContext, email: string) {
  const key = createHash("sha256").update(email).digest("hex");
  try {
    const slot = await claimWindowSlot(
      context,
      `sign-in-code-email:${key}`,
      CODE_EMAILS_PER_ADDRESS,
    );
    return slot !== null;
  } catch (error) {
    console.error("Failed to check the sign-in code email limit", {
      error: error instanceof Error ? error.message : String(error),
    });
    return true;
  }
}

async function sendCode(pending: PendingSignIn, code: string) {
  try {
    await sendAuthEmail({
      to: pending.email,
      type: "sign_in_code",
      code,
      provider: pending.providerId,
    });
    return true;
  } catch (error) {
    console.error("Failed to send a sign-in code", {
      error: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}

function newCode() {
  return randomInt(0, 10 ** SIGN_IN_CODE_LENGTH)
    .toString()
    .padStart(SIGN_IN_CODE_LENGTH, "0");
}

// Keyed by the sign-in's id as well, so a code means nothing for another one.
function hashCode(secret: string, id: string, code: string) {
  return createHmac("sha256", secret).update(`${id}:${code}`).digest("hex");
}

function codeMatches(secret: string, id: string, code: string, hash: string) {
  if (!new RegExp(`^[0-9]{${SIGN_IN_CODE_LENGTH}}$`).test(code)) return false;
  const expected = Buffer.from(hash, "hex");
  const actual = Buffer.from(hashCode(secret, id, code), "hex");
  return expected.length === actual.length && timingSafeEqual(expected, actual);
}

function secondsUntilResend(record: PendingRecord) {
  const due = record.sentAt + SIGN_IN_CODE_RESEND_SECONDS * 1000;
  return Math.max(0, Math.ceil((due - Date.now()) / 1000));
}

interface ProviderEntry {
  id: string;
  options?: { disableSignUp?: boolean };
  accountSubject?: (input: {
    tokens: object;
    profile: Record<string, unknown>;
  }) => unknown;
}

async function findProvider(ctx: GenericEndpointContext, providerId: string) {
  for (const entry of ctx.context.socialProviders as unknown[]) {
    const provider = (
      typeof entry === "function" ? await entry() : entry
    ) as ProviderEntry;
    if (provider?.id === providerId) return provider;
  }
  return null;
}

// The account key Better Auth gives the provider's account: the same
// `accountSubject` its callback uses, so finishing with the code links exactly
// the account the provider signed in.
async function accountIdOf(
  ctx: GenericEndpointContext,
  providerId: string,
  profile: Record<string, unknown> | undefined,
) {
  const provider = await findProvider(ctx, providerId);
  if (!profile || !provider?.accountSubject) return null;
  try {
    const subject = await provider.accountSubject({ tokens: {}, profile });
    const id =
      typeof subject === "string" || typeof subject === "number"
        ? String(subject)
        : "";
    return id.trim() ? id : null;
  } catch {
    return null;
  }
}

function isCodeProvider(
  providerId: string,
): providerId is PendingSignIn["providerId"] {
  return CODE_PROVIDERS.has(providerId);
}

function isNoticeProvider(
  providerId: string,
): providerId is SignInEmailProvider {
  return NOTICE_PROVIDERS.has(providerId);
}

function parseJSON(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}
