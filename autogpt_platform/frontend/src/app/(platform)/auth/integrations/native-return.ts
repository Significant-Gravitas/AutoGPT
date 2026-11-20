import { NextResponse } from "next/server";
import { environment } from "@/services/environment";

// Matches NATIVE_STATE_PREFIX in backend/integrations/credentials_store.py.
const NATIVE_STATE_PREFIX = "native.";

// The app registers one scheme and waits for exactly this address
// (OAUTH_RETURN_URL in mobile-app/src/lib/api/integrations.ts). Production
// answers the store app; every other deployment answers AutoGPT Dev
// (mobile-app/app.config.ts). The scheme comes from this deployment's own
// environment and never from the request, so a code can't be sent elsewhere.
function getNativeReturnUrl() {
  const isProduction = environment.isProd() && !environment.isVercelPreview();
  const scheme = isProduction ? "autogpt-app" : "autogpt-app-dev";
  return `${scheme}://integrations/oauth`;
}

interface NativeCallback {
  state: string;
  code: string | null;
  iss?: string | null;
  error: string | null;
  errorDescription: string | null;
}

export function isNativeState(state: string | null): state is string {
  return state?.startsWith(NATIVE_STATE_PREFIX) ?? false;
}

export function redirectToNativeApp({
  state,
  code,
  iss,
  error,
  errorDescription,
}: NativeCallback) {
  const url = new URL(getNativeReturnUrl());
  const params = code
    ? { code, state, iss }
    : {
        state,
        error: error || "missing_code",
        error_description: errorDescription,
      };
  for (const [key, value] of Object.entries(params)) {
    if (value) url.searchParams.set(key, value);
  }
  return NextResponse.redirect(url, {
    status: 302,
    headers: { "Cache-Control": "no-store", "Referrer-Policy": "no-referrer" },
  });
}
