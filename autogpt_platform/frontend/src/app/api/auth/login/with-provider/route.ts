import { auth } from "@/lib/auth/auth";
import { connectAtSignInScopes } from "@/lib/auth/connect-at-sign-in";
import { sanitizeAuthNext } from "@/lib/auth-redirect";
import { LoginProvider } from "@/types/auth";
import { APIError } from "better-auth/api";
import { NextResponse } from "next/server";
import { isWaitlistError, logWaitlistError } from "../../utils";

export async function POST(request: Request) {
  let body: {
    provider?: LoginProvider;
    redirectTo?: string;
    connect?: boolean;
  };
  try {
    body = await request.json();
  } catch {
    // A malformed body is the caller's mistake, not ours — reporting 500 would
    // put client syntax errors in our server-error telemetry and invite retries.
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const provider: LoginProvider | undefined = body?.provider;
    const redirectTo: string | undefined = body?.redirectTo;

    if (!provider) {
      return NextResponse.json({ error: "Invalid provider" }, { status: 400 });
    }

    try {
      const callbackURL =
        redirectTo || process.env.AUTH_CALLBACK_URL || "/auth/callback";
      // Google and Microsoft are also asked for the data scopes when their
      // sign-in can connect (connect-at-sign-in.ts), unless the person chose
      // to sign in without connecting.
      const scopes =
        body.connect === false ? null : await connectAtSignInScopes(provider);
      const { url } = await auth.api.signInSocial({
        body: {
          provider,
          callbackURL,
          // A sign-in refused at the provider callback (a banned account; for
          // Microsoft also an email Microsoft did not confirm, or an address
          // that belongs to another login) comes back to the login page with
          // the error code in the query.
          errorCallbackURL: loginPageFor(
            provider,
            callbackURL,
            scopes !== null,
          ),
          ...(scopes ? { scopes: [...scopes] } : {}),
        },
        headers: request.headers,
      });

      // The page notes a connecting sign-in, to offer one without
      // connecting if the person comes back without finishing it.
      return NextResponse.json({ url, connecting: scopes !== null });
    } catch (error) {
      if (error instanceof APIError) {
        // Match on the body message ("Signups are not allowed."), not
        // error.message — the latter is the status ("FORBIDDEN"), which never
        // matches the waitlist patterns (same trap as the email signup path).
        if (isWaitlistError(error.body?.code, error.body?.message)) {
          logWaitlistError("OAuth Provider", error.message);
          return NextResponse.json({ error: "not_allowed" }, { status: 403 });
        }

        // Preserve Better Auth's status so callers can tell throttling (429)
        // or auth failure (401) apart from a bad request.
        return NextResponse.json(
          { error: error.body?.message || error.message },
          { status: error.statusCode ?? 400 },
        );
      }
      throw error;
    }
  } catch {
    return NextResponse.json(
      { error: "Failed to initiate OAuth" },
      { status: 500 },
    );
  }
}

// The login page, with the callback's `next`, so a deep link survives
// logging in another way. A Microsoft refusal says it is one: the page
// explains Microsoft's codes in Microsoft's words (microsoft-login.ts). A
// sign-in that also asked for the data scopes says so too, so the page can
// offer to sign in without them: a company's admin may block them.
function loginPageFor(
  provider: LoginProvider,
  callbackURL: string,
  connecting: boolean,
) {
  const params = new URLSearchParams();
  if (provider === "microsoft") params.set("provider", provider);
  if (connecting) params.set("connect", provider);
  const next = nextOf(callbackURL);
  if (next) params.set("next", next);
  const query = params.toString();
  return query ? `/login?${query}` : "/login";
}

function nextOf(callbackURL: string) {
  try {
    return sanitizeAuthNext(
      new URL(callbackURL, "http://localhost").searchParams.get("next"),
    );
  } catch {
    return null;
  }
}
