import { useState } from "react";

import { authClient } from "@/lib/auth/client";

interface SignInError {
  code?: string;
  message?: string;
  status?: number;
}

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
      const result = await authClient.signIn.email({
        email: email.trim(),
        password,
      });
      if (result.error) setError(errorCopy(result.error));
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

function errorCopy(error: SignInError | null) {
  if (error?.code === "EMAIL_NOT_VERIFIED") {
    return "Verify your email first. Open the link we sent you, then sign in here.";
  }
  if (!error?.status) {
    return "Can't reach AutoGPT. Check your connection and try again.";
  }
  if (error.status === 401) return "Wrong email or password.";
  if (error.code?.endsWith("_ORIGIN")) {
    return "You can't sign in right now. Try again later.";
  }
  return error.message || "You can't sign in right now. Try again later.";
}
