import { NextResponse } from "next/server";

// Matches NATIVE_STATE_PREFIX in backend/integrations/credentials_store.py.
const NATIVE_STATE_PREFIX = "native.";
const DEFAULT_NATIVE_RETURN_URL = "autogpt-app://integrations/oauth";

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
  const url = nativeReturnURL();
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

function nativeReturnURL() {
  const configured = process.env.NATIVE_OAUTH_RETURN_URL;
  if (configured && URL.canParse(configured)) {
    const url = new URL(configured);
    if (url.protocol !== "http:" && url.protocol !== "https:") return url;
  }
  return new URL(DEFAULT_NATIVE_RETURN_URL);
}
