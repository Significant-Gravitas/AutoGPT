import { useState } from "react";

import { signInWithEmail } from "@/lib/auth/nativeSession";
import {
  NO_APP_SESSION_CODE,
  type SignInFailure,
} from "@/lib/auth/sessionHelpers";
import { whenSignedOut } from "@/lib/session";

export function useLoginScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const canSubmit =
    email.trim().length > 0 && password.length > 0 && !isSubmitting;

  async function submit() {
    if (!canSubmit) return;
    setError(null);
    setIsSubmitting(true);
    try {
      // The last sign-out clears the cache again when it ends, which would
      // take this sign-in's data with it.
      await whenSignedOut();
      const failure = await signInWithEmail(email.trim(), password);
      if (failure) setError(errorCopy(failure));
    } catch {
      setError(errorCopy(null));
    } finally {
      setIsSubmitting(false);
    }
  }

  return {
    email,
    setEmail,
    password,
    setPassword,
    error,
    isSubmitting,
    canSubmit,
    submit,
  };
}

function errorCopy(error: SignInFailure | null) {
  if (error?.code === "EMAIL_NOT_VERIFIED") {
    return "Verify your email first. Open the link we sent you, then sign in here.";
  }
  if (!error?.status) {
    return "Can't reach AutoGPT. Check your connection and try again.";
  }
  if (error.status === 401) return "Wrong email or password.";
  if (error.code === NO_APP_SESSION_CODE) {
    return "This server can't sign the app in yet. Update it and try again.";
  }
  return error.message || "You can't sign in right now. Try again later.";
}
