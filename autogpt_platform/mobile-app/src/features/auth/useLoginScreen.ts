import { useState } from "react";

import { signInWithEmail } from "@/lib/auth/nativeSession";
import { whenSignedOut } from "@/lib/session";

import { signInErrorCopy } from "./signupHelpers";

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
      if (failure) setError(signInErrorCopy(failure));
    } catch {
      setError(signInErrorCopy(null));
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
