import { postV1GetOrCreateUser } from "@/app/api/__generated__/endpoints/auth/auth";
import { getOnboardingStatus } from "@/app/api/helpers";
import { sanitizeAuthNext } from "@/lib/auth-redirect";
import {
  EMAIL_VERIFICATION_NOTICE_PARAM,
  type EmailVerificationNotice,
  hasMarketingOptOutParam,
} from "@/lib/auth/email-verification";
import { getServerSession } from "@/lib/auth/server/getServerSession";
import { recordSignupConsent } from "@/lib/auth/server/recordSignupConsent";
import { rollbackSession } from "@/lib/auth/server/rollbackSession";
import { type SignupMethod } from "@/services/analytics/account-created-cookie";
import { markAccountCreated } from "@/services/analytics/account-created-server";
import {
  scheduleAccountCreatedGoal,
  wasAccountCreated,
} from "@/services/analytics/datafast-server";
import { takeMarketingOptOutFlag } from "@/services/analytics/marketing-opt-out-server";
import { revalidatePath } from "next/cache";
import { NextResponse } from "next/server";

// Post-OAuth landing page, not the OAuth exchange. Better Auth's built-in
// /api/auth/callback/{provider} does the code exchange and sets the session
// cookie, then redirects here because this is the `callbackURL` we hand it in
// /api/auth/login/with-provider. So by the time we run, the session already
// exists and we only provision the backend user and decide where to send them.
//
// An email verification link lands here too (`?method=email`, see
// lib/auth/email-verification.ts) after /api/auth/verify-email has signed the
// user in, so a verified email sign-up is provisioned and counted here, once.
function getPublicOrigin(requestOrigin: string) {
  const configuredURL =
    process.env.BETTER_AUTH_URL || process.env.NEXT_PUBLIC_FRONTEND_BASE_URL;
  if (!configuredURL) return requestOrigin;

  try {
    const url = new URL(configuredURL);
    return url.protocol === "http:" || url.protocol === "https:"
      ? url.origin
      : requestOrigin;
  } catch {
    return requestOrigin;
  }
}

function getSignupMethod(searchParams: URLSearchParams): SignupMethod {
  return searchParams.get("method") === "email" ? "email" : "google";
}

// Better Auth redirects here without a session when the link is expired,
// invalid or already used (`?error=`), and when the address was already
// verified. Either way the user can log in: an unverified account is sent a
// fresh link from the login form.
function getEmailVerificationLoginPath(searchParams: URLSearchParams) {
  const notice: EmailVerificationNotice = searchParams.get("error")
    ? "expired"
    : "verified";
  const params = new URLSearchParams({
    [EMAIL_VERIFICATION_NOTICE_PARAM]: notice,
  });
  const next = sanitizeAuthNext(searchParams.get("next"));
  if (next) params.set("next", next);
  return `/login?${params.toString()}`;
}

export async function GET(request: Request) {
  const { searchParams, origin: requestOrigin } = new URL(request.url);
  const publicOrigin = getPublicOrigin(requestOrigin);
  const signupMethod = getSignupMethod(searchParams);

  let next = "/copilot";

  const session = await getServerSession();

  if (session?.user) {
    try {
      const createUserResponse = await postV1GetOrCreateUser();
      // Consumed once the user exists, new or returning, so it applies to
      // this sign-in only. Not taken before provisioning succeeds: a failure
      // redirects to /error and the retry must still carry the refusal.
      // Never throws, so it can't reach the rollback below.
      const cookieOptOut = await takeMarketingOptOutFlag();
      const accountCreated = wasAccountCreated(createUserResponse);
      // An email sign-up's refusal comes in its verification link. Only the
      // account the link creates takes it, so a crafted link can't change an
      // existing account.
      const marketingOptOut =
        cookieOptOut ||
        (accountCreated &&
          signupMethod === "email" &&
          hasMarketingOptOutParam(searchParams));
      if (accountCreated) {
        await scheduleAccountCreatedGoal(signupMethod);
        await markAccountCreated(signupMethod);
      }
      // A returning account that opted out on /signup before continuing with
      // Google records the refusal too: the page has already told them they
      // won't get marketing emails. Never throws, so a failed consent write
      // can't reach the rollback below or change where the user lands.
      if (accountCreated || marketingOptOut) {
        await recordSignupConsent({
          userID: session.user.id,
          marketingOptOut,
        });
      }

      const { shouldShowOnboarding } = await getOnboardingStatus();
      // Prefer a sanitized ?next (relative paths only — sanitizeAuthNext drops
      // absolute and protocol-relative values, so a crafted ?next can't
      // open-redirect the user off-site); otherwise route by onboarding state.
      // Resolve the final target BEFORE revalidating so we revalidate the page
      // the user actually lands on.
      next =
        sanitizeAuthNext(searchParams.get("next")) ??
        (shouldShowOnboarding ? "/onboarding" : "/copilot");
      revalidatePath(next, "layout");
    } catch (createUserError) {
      console.error("Error creating user:", createUserError);

      // Better Auth already set the session cookie before redirecting here, so
      // a provisioning failure would otherwise leave the browser "logged in"
      // with no backend user. Revoke the session to match login/signup, which
      // both rollbackSession on the same failure.
      await rollbackSession();

      // Handle ApiError from the backend API client
      if (
        createUserError &&
        typeof createUserError === "object" &&
        "status" in createUserError
      ) {
        const apiError = createUserError as { status: number };

        if (apiError.status === 401) {
          // Authentication issues - token missing/invalid
          return NextResponse.redirect(
            `${publicOrigin}/error?message=auth-token-invalid`,
          );
        } else if (apiError.status >= 500) {
          // Server/database errors
          return NextResponse.redirect(
            `${publicOrigin}/error?message=server-error`,
          );
        } else if (apiError.status === 429) {
          // Rate limiting
          return NextResponse.redirect(
            `${publicOrigin}/error?message=rate-limited`,
          );
        }
      }

      // Handle network/fetch errors
      if (
        createUserError instanceof TypeError &&
        createUserError.message.includes("fetch")
      ) {
        return NextResponse.redirect(
          `${publicOrigin}/error?message=network-error`,
        );
      }

      // Generic user creation failure
      return NextResponse.redirect(
        `${publicOrigin}/error?message=user-creation-failed`,
      );
    }

    return NextResponse.redirect(`${publicOrigin}${next}`);
  }

  if (signupMethod === "email") {
    return NextResponse.redirect(
      `${publicOrigin}${getEmailVerificationLoginPath(searchParams)}`,
    );
  }

  // return the user to an error page with instructions
  return NextResponse.redirect(`${publicOrigin}/auth/auth-code-error`);
}
