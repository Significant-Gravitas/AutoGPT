import { useRef, useState } from "react";

import { provisionNewAccount } from "@/lib/api/account";
import {
  resendVerification,
  signInWithEmail,
  signUpWithEmail,
} from "@/lib/auth/nativeSession";
import { verificationCallbackURL } from "@/lib/auth/sessionHelpers";
import { signOut, whenSignedOut } from "@/lib/session";
import { useMountEffect } from "@/lib/useMountEffect";

import {
  RESEND_COOLDOWN_SECONDS,
  canSubmitSignup,
  passwordProblem,
  signupErrorCopy,
  verifyErrorCopy,
} from "./signupHelpers";

export function useSignupScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [optedOut, setOptedOut] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  // Set once the server has emailed a link: shows Check your inbox.
  const [pendingEmail, setPendingEmail] = useState<string | null>(null);
  const [cooldown, setCooldown] = useState(0);
  const [isResending, setIsResending] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const countdown = useRef<ReturnType<typeof setInterval> | null>(null);

  const canSubmit = canSubmitSignup(email, password) && !isSubmitting;

  function stopCountdown() {
    if (countdown.current) clearInterval(countdown.current);
    countdown.current = null;
  }

  function startCountdown() {
    stopCountdown();
    const endsAt = Date.now() + RESEND_COOLDOWN_SECONDS * 1000;
    setCooldown(RESEND_COOLDOWN_SECONDS);
    countdown.current = setInterval(() => {
      const left = Math.max(0, Math.ceil((endsAt - Date.now()) / 1000));
      setCooldown(left);
      if (left === 0) stopCountdown();
    }, 1000);
  }

  useMountEffect(() => stopCountdown);

  function toggleOptOut() {
    setOptedOut((value) => !value);
  }

  async function submit() {
    if (!canSubmit) return;
    const problem = passwordProblem(password);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    setIsSubmitting(true);
    try {
      // The last sign-out clears the cache again when it ends, which would
      // take this sign-in's data with it.
      await whenSignedOut();
      const result = await signUpWithEmail({
        email,
        password,
        marketingOptOut: optedOut,
      });
      if (result.kind === "verify") setPendingEmail(email.trim());
      else if (result.kind === "refused") setError(signupErrorCopy(result.failure));
      else await provision();
    } catch {
      setError(signupErrorCopy(null));
    } finally {
      setIsSubmitting(false);
    }
  }

  // Signed in at once: the root layout has already left this screen, so a
  // failure can't be shown here. Signing out puts the person back at the start.
  async function provision() {
    try {
      await provisionNewAccount(optedOut);
    } catch {
      await signOut();
    }
  }

  async function confirmOpened() {
    if (!pendingEmail || isChecking) return;
    setError(null);
    setIsChecking(true);
    try {
      await whenSignedOut();
      const failure = await signInWithEmail(
        pendingEmail,
        password,
        verificationCallbackURL(optedOut),
      );
      if (failure) setError(verifyErrorCopy(failure));
    } catch {
      setError(verifyErrorCopy(null));
    } finally {
      setIsChecking(false);
    }
  }

  async function resend() {
    if (!pendingEmail || isResending || cooldown > 0) return;
    setError(null);
    setIsResending(true);
    try {
      const failure = await resendVerification({
        email: pendingEmail,
        marketingOptOut: optedOut,
      });
      if (failure) setError(verifyErrorCopy(failure));
      else startCountdown();
    } catch {
      setError(verifyErrorCopy(null));
    } finally {
      setIsResending(false);
    }
  }

  function startAgain() {
    stopCountdown();
    setCooldown(0);
    setPendingEmail(null);
    setEmail("");
    setError(null);
  }

  return {
    email,
    setEmail,
    password,
    setPassword,
    optedOut,
    toggleOptOut,
    error,
    isSubmitting,
    canSubmit,
    submit,
    pendingEmail,
    cooldown,
    isResending,
    isChecking,
    confirmOpened,
    resend,
    startAgain,
  };
}
