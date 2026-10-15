import { type AuthEmailContext, claimEmailSlot } from "./auth-email-cooldown";
import { currentAuthEndpoint } from "./auth-endpoint";
import { sendAuthEmail } from "./email";
import { hasAccountBeenUsed, sendSetPasswordLink } from "./set-password-link";

interface Args {
  user: { id: string; email: string };
  url: string;
  request?: Request;
  getAuthContext: () => Promise<AuthEmailContext>;
  hasPlatformUser: (userId: string) => Promise<boolean>;
  requireEmailVerification: boolean;
  resetRedirectTo: string;
}

/**
 * Better Auth's sendVerificationEmail. Sign-in sends one every time an
 * unverified account signs in, and the login page calls auth.api directly,
 * which Better Auth's rate limiter never sees, so whoever set the password
 * could have us mail the address without limit. Sign-in and sign-up therefore
 * share one email per address per window. If the cooldown can't be checked,
 * the email still goes: a verification link matters more than the cap. With
 * verification off, neither sends one. Nothing else is capped, flag or no
 * flag: a change of email sends to the new address, and a retry must not be
 * dropped while the page reports success.
 *
 * The resend button's own route skips that cap (it is capped per IP instead,
 * see ip-email-cap.ts) because its answer has to say whether the email went.
 * With verification required, its link would sign in whoever opens it, so for
 * a used account, whose password someone other than the address owner may
 * hold, it sends the set-password link a repeat sign-up gets instead.
 */
export async function sendVerificationLink(args: Args) {
  const { user, url, request, getAuthContext } = args;
  if (isResendRequest(request)) {
    if (args.requireEmailVerification && (await sendsSetPasswordLink(args)))
      return;
  } else if (args.requireEmailVerification && (await isSignUpOrSignIn())) {
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

async function sendsSetPasswordLink(args: Args) {
  // No context means no set-password link either, and falling back to the
  // verification link would sign in whoever opens it, so this one errors.
  const context = await args.getAuthContext();
  // Unknown counts as used: the set-password link is the safe one to send.
  const used = await hasAccountBeenUsed(
    context,
    args.user.id,
    args.hasPlatformUser,
    true,
  );
  if (!used) return false;
  await sendSetPasswordLink(context, args.user, args.resetRedirectTo);
  return true;
}

// Read from Better Auth's endpoint context: the login page calls auth.api
// directly, with no request.
async function isSignUpOrSignIn() {
  const { path } = await currentAuthEndpoint();
  return path === "/sign-up/email" || path === "/sign-in/email";
}

function isResendRequest(request: Request | undefined) {
  if (!request) return false;
  return new URL(request.url).pathname.endsWith("/send-verification-email");
}
