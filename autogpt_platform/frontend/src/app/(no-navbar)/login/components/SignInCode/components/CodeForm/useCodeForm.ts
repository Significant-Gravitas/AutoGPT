import { providerCallbackPath } from "@/lib/auth/provider-callback";
import { type FormEvent, useEffect, useState } from "react";
import {
  SIGN_IN_CODE_COPY,
  SIGN_IN_CODE_INPUT_ID,
  type SignInCodeEnd,
  submitSignInCode,
} from "../../helpers";

interface Args {
  next: string | null;
  onEnded: (end: SignInCodeEnd) => void;
  onWaitlisted: () => void;
}

export function useCodeForm({ next, onEnded, onWaitlisted }: Args) {
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);

  useEffect(() => {
    document.getElementById(SIGN_IN_CODE_INPUT_ID)?.focus();
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!code.trim() || isVerifying) return;
    setIsVerifying(true);
    setError(null);
    const outcome = await submitSignInCode(code).catch(() => null);
    if (outcome?.kind === "signed-in") {
      // A full navigation, as after any provider sign-in, so the new
      // session's cookie reaches the callback page.
      window.location.href = providerCallbackPath(outcome.provider, next);
      return;
    }
    setIsVerifying(false);
    if (outcome?.kind === "ended") onEnded(outcome.end);
    else if (outcome?.kind === "waitlisted") onWaitlisted();
    else if (outcome?.kind === "invalid") setError(SIGN_IN_CODE_COPY.invalid);
    else setError(outcome?.message ?? SIGN_IN_CODE_COPY.failed);
  }

  return {
    code,
    error,
    isVerifying,
    handleCodeChange: (value: string) => {
      setCode(value);
      setError(null);
    },
    handleSubmit,
  };
}
