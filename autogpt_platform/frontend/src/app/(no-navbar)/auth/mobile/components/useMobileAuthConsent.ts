"use client";

import { useState } from "react";
import { MOBILE_AUTH_CALLBACK } from "@/lib/auth/mobile-auth-helpers";
import { readMobileCallback } from "./helpers";

export function useMobileAuthConsent(
  codeChallenge: string,
  state: string,
  userID: string,
) {
  const [isConnecting, setIsConnecting] = useState(false);
  const [callbackURL, setCallbackURL] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function connect() {
    if (isConnecting) return;
    setIsConnecting(true);
    setError(null);
    try {
      const response = await fetch("/api/auth/mobile/authorize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({
          code_challenge: codeChallenge,
          state,
          expected_user_id: userID,
        }),
      });
      if (!response.ok) throw new Error("Authorization failed");
      const callback = readMobileCallback(await response.json(), state);
      if (!callback) throw new Error("Invalid authorization callback");
      setCallbackURL(callback);
      window.location.assign(callback);
    } catch {
      setError(
        "Could not connect AutoGPT. Please try again or restart sign-in in the app.",
      );
    } finally {
      setIsConnecting(false);
    }
  }

  function returnToApp() {
    if (callbackURL) window.location.assign(callbackURL);
  }

  function cancel() {
    const callback = new URL(MOBILE_AUTH_CALLBACK);
    callback.searchParams.set("error", "access_denied");
    callback.searchParams.set("state", state);
    window.location.assign(callback.toString());
  }

  return { isConnecting, callbackURL, error, connect, returnToApp, cancel };
}
