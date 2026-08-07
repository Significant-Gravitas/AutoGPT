import { postV1GetOrCreateUser } from "@/app/api/__generated__/endpoints/auth/auth";
import { getOnboardingStatus } from "@/app/api/helpers";
import { sanitizeAuthNext } from "@/lib/auth-redirect";
import { getServerSession } from "@/lib/auth/server/getServerSession";
import { rollbackSession } from "@/lib/auth/server/rollbackSession";
import {
  scheduleAccountCreatedGoal,
  wasAccountCreated,
} from "@/services/analytics/datafast-server";
import { revalidatePath } from "next/cache";
import { NextResponse } from "next/server";

// Post-OAuth landing page, not the OAuth exchange. Better Auth's built-in
// /api/auth/callback/{provider} does the code exchange and sets the session
// cookie, then redirects here because this is the `callbackURL` we hand it in
// /api/auth/login/with-provider. So by the time we run, the session already
// exists and we only provision the backend user and decide where to send them.
function httpOrigin(value: string | null) {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:"
      ? url.origin
      : null;
  } catch {
    return null;
  }
}

function getPublicOrigin(request: Request, requestOrigin: string) {
  const configuredURL =
    process.env.BETTER_AUTH_URL || process.env.NEXT_PUBLIC_FRONTEND_BASE_URL;
  const configuredOrigin = httpOrigin(configuredURL || null);
  if (configuredOrigin) return configuredOrigin;

  const forwardedHost = request.headers
    .get("x-forwarded-host")
    ?.split(",", 1)[0]
    ?.trim();
  const forwardedProto = request.headers
    .get("x-forwarded-proto")
    ?.split(",", 1)[0]
    ?.trim();
  if (!forwardedHost) return requestOrigin;
  try {
    const protocol =
      forwardedProto === "http" || forwardedProto === "https"
        ? forwardedProto
        : new URL(requestOrigin).protocol.slice(0, -1);
    const forwardedURL = new URL(`${protocol}://${forwardedHost}`);
    if (
      forwardedURL.username ||
      forwardedURL.password ||
      forwardedURL.pathname !== "/" ||
      forwardedURL.search ||
      forwardedURL.hash
    ) {
      return requestOrigin;
    }
    return forwardedURL.origin;
  } catch {
    return requestOrigin;
  }
}

export async function GET(request: Request) {
  const { searchParams, origin: requestOrigin } = new URL(request.url);
  const publicOrigin = getPublicOrigin(request, requestOrigin);

  let next = "/copilot";

  const session = await getServerSession();

  if (session?.user) {
    try {
      const createUserResponse = await postV1GetOrCreateUser();
      if (wasAccountCreated(createUserResponse)) {
        await scheduleAccountCreatedGoal("google");
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

  // return the user to an error page with instructions
  return NextResponse.redirect(`${publicOrigin}/auth/auth-code-error`);
}
