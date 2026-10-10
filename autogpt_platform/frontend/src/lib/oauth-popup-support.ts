const OAUTH_NATIVE_BROWSER_REQUIRED =
  "To use browser sign-in for this connection, choose Open in browser from the app menu. Sign in to the same AutoGPT account, connect the service there, then return to the app and choose Reload.";

export class NativeOAuthPopupError extends Error {
  constructor() {
    super(OAUTH_NATIVE_BROWSER_REQUIRED);
    this.name = "NativeOAuthPopupError";
  }
}

export function isNativeAutoGPTApp() {
  return (
    typeof navigator !== "undefined" &&
    /(?:^|\s)AutoGPTMobile\/(?:iOS|Android)\s*$/.test(navigator.userAgent)
  );
}

export function assertOAuthPopupSupported() {
  if (isNativeAutoGPTApp()) throw new NativeOAuthPopupError();
}
