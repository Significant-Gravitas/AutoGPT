import type { GenericEndpointContext } from "@better-auth/core";
import type {
  ValidateUserInfoResult,
  ValidateUserInfoSource,
} from "better-auth";
import { CODE_PROVIDER_IDS, providerRunsMailbox } from "./mailbox-provider";
import { ACCOUNT_NOT_LINKED_CODE } from "./microsoft-login";
import {
  SIGN_IN_CODE_LOCKED,
  SIGN_IN_CODE_SENT,
  SIGN_IN_CODE_UNSENT,
} from "./sign-in-code";
import { hashCode, newCode, newPendingID } from "./sign-in-code-crypto";
import {
  claimCodeEmail,
  releaseCodeEmail,
  sendCode,
} from "./sign-in-code-emails";
import { setPendingCookie } from "./sign-in-code-cookie";
import { isLockedOut } from "./sign-in-code-lockout";
import {
  forgetPending,
  type PendingSignIn,
  storePending,
} from "./sign-in-code-store";
import { accountIDOf, providerConfirmedByCode } from "./sign-in-provider";

/**
 * A provider sign-in that would join an existing account finishes with a code
 * emailed to the address, typed into the page that started it, unless the
 * provider runs that mailbox (mailbox-provider.ts). Better Auth would
 * otherwise link any provider that says it verified the address, and a
 * provider that vouched for addresses it never verified would open every
 * account that uses them.
 *
 * An address the provider didn't verify joins nothing and creates nothing:
 * anyone can open such a provider account with someone else's address, so a
 * code for it would only invite guessing. Better Auth refuses those links
 * before this hook runs; it refuses them here too.
 *
 * The code can't be a link: the mailbox's owner could then approve someone
 * else's sign-in with a click. It is checked against the browser that started
 * the sign-in (a signed cookie), and finishing it runs Better Auth's own
 * sign-in with the provider account the callback saw, so every session hook
 * still applies (sign-in-code-plugin.ts).
 *
 * Apple keeps Better Auth's rule: Apple verifies every address it hands out,
 * and its native sign-in has no page to type a code into.
 *
 * This is Better Auth's `user.validateUserInfo`. Returning an error stops the
 * sign-in and sends the browser to its error URL with that code: /login,
 * which then asks for the code.
 */
export async function requireSignInCode(
  data: { user: Record<string, unknown>; source: ValidateUserInfoSource },
  ctx: GenericEndpointContext,
): Promise<ValidateUserInfoResult | undefined> {
  const { user, source } = data;
  if (source.method !== "oauth" || !source.oauth) return;
  if (source.action !== "link-account") return;
  const { providerId, profile } = source.oauth;
  if (!isCodeProvider(providerId)) return;
  // The code's own endpoint, finishing the sign-in it was emailed for.
  if (providerConfirmedByCode(ctx) === providerId) return;
  const email = typeof user.email === "string" ? user.email.toLowerCase() : "";
  if (!email || user.emailVerified !== true) {
    return { error: ACCOUNT_NOT_LINKED_CODE };
  }
  const claim = { email, emailVerified: true, profile };
  if (providerRunsMailbox(providerId, claim)) return;
  if (await isLockedOut(ctx.context, email)) {
    return { error: SIGN_IN_CODE_LOCKED };
  }
  const accountId = await accountIDOf(ctx, providerId, profile);
  if (!accountId) return { error: SIGN_IN_CODE_UNSENT };
  return startSignInCode(ctx, {
    providerId,
    accountId,
    email,
    name: typeof user.name === "string" ? user.name : "",
    image: typeof user.image === "string" ? user.image : null,
  });
}

async function startSignInCode(
  ctx: GenericEndpointContext,
  pending: PendingSignIn,
): Promise<ValidateUserInfoResult> {
  const claim = await claimCodeEmail(ctx.context, pending.email);
  if ("refusal" in claim) return { error: claim.refusal };
  const id = newPendingID();
  const code = newCode();
  const stored = await storePending(ctx.context, id, {
    ...pending,
    codeHash: hashCode(ctx.context.secret, id, code),
    sentAt: Date.now(),
  }).then(
    () => true,
    (error: unknown) => {
      console.error("Failed to store a waiting sign-in", {
        error: error instanceof Error ? error.message : String(error),
      });
      return false;
    },
  );
  if (!stored) {
    await releaseCodeEmail(ctx.context, claim.slot);
    return { error: SIGN_IN_CODE_UNSENT };
  }
  if (!(await sendCode(ctx.context, pending, code, claim.slot))) {
    await forgetPending(ctx.context, id);
    return { error: SIGN_IN_CODE_UNSENT };
  }
  await setPendingCookie(ctx, id);
  return { error: SIGN_IN_CODE_SENT };
}

function isCodeProvider(
  providerId: string,
): providerId is PendingSignIn["providerId"] {
  return (CODE_PROVIDER_IDS as readonly string[]).includes(providerId);
}
