import { useToast } from "@/components/molecules/Toast/use-toast";
import { providerCallbackPath } from "@/lib/auth/provider-callback";
import { signInProviderName } from "@/lib/auth/sign-in-code";
import type { LoginProvider } from "@/types/auth";
import { useQuery } from "@tanstack/react-query";
import { type FormEvent, useEffect, useState } from "react";
import {
  fetchPendingSignInCode,
  resendSignInCode,
  SIGN_IN_CODE_COPY,
  SIGN_IN_CODE_INPUT_ID,
  submitSignInCode,
} from "./helpers";

interface Args {
  next: string | null;
}

type Screen = "code" | "expired" | "waitlisted";

export function useSignInCode({ next }: Args) {
  const { toast } = useToast();
  const status = useQuery({
    queryKey: ["auth", "sign-in-code"],
    queryFn: fetchPendingSignInCode,
    staleTime: Infinity,
    retry: false,
  });
  const pending = status.data ?? null;
  const [screen, setScreen] = useState<Screen>("code");
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [isResending, setIsResending] = useState(false);
  const [cooldown, setCooldown] = useState(0);

  useEffect(() => {
    if (pending) setCooldown(pending.resendInSeconds);
  }, [pending]);

  useEffect(() => {
    if (pending) document.getElementById(SIGN_IN_CODE_INPUT_ID)?.focus();
  }, [pending]);

  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setTimeout(() => setCooldown((value) => value - 1), 1000);
    return () => clearTimeout(timer);
  }, [cooldown]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!code.trim() || isVerifying) return;
    setIsVerifying(true);
    setError(null);
    try {
      const outcome = await submitSignInCode(code);
      if (outcome.kind === "signed-in") {
        // A full navigation, as after any provider sign-in, so the new
        // session's cookie reaches the callback page.
        window.location.href = providerCallbackPath(
          outcome.provider as LoginProvider,
          next,
        );
        return;
      }
      if (outcome.kind === "invalid") setError(SIGN_IN_CODE_COPY.invalid);
      else if (outcome.kind === "expired") setScreen("expired");
      else if (outcome.kind === "waitlisted") setScreen("waitlisted");
      else setError(outcome.message);
    } catch {
      setError(SIGN_IN_CODE_COPY.failed);
    }
    setIsVerifying(false);
  }

  async function handleResend() {
    setIsResending(true);
    try {
      const outcome = await resendSignInCode();
      if (outcome.kind === "sent") {
        setCooldown(outcome.resendInSeconds);
        setCode("");
        setError(null);
        toast({
          title: pending
            ? `New code sent to ${pending.email}`
            : "New code sent",
          variant: "success",
        });
      } else if (outcome.kind === "expired") {
        setScreen("expired");
      } else {
        toast({ title: outcome.message, variant: "destructive" });
      }
    } catch {
      toast({ title: SIGN_IN_CODE_COPY.failed, variant: "destructive" });
    } finally {
      setIsResending(false);
    }
  }

  return {
    // Nothing waits on a code here: it was finished, ran out, or never
    // started in this browser. A lookup that failed still lets the code be
    // typed: the code itself is checked either way.
    screen: status.isSuccess && !pending ? "expired" : screen,
    isLoading: status.isPending,
    email: pending?.email ?? null,
    providerName: pending ? signInProviderName(pending.provider) : null,
    code,
    error,
    isVerifying,
    isResending,
    cooldown,
    canResend: cooldown <= 0 && !isResending,
    handleCodeChange: (value: string) => {
      setCode(value);
      setError(null);
    },
    handleSubmit,
    handleResend,
  };
}
