import {
  SIGN_IN_CODE_EXPIRED,
  signInProviderName,
} from "@/lib/auth/sign-in-code";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { fetchPendingSignInCode, type SignInCodeEnd } from "./helpers";

type Screen = "code" | "waitlisted" | SignInCodeEnd;

export function useSignInCode() {
  const status = useQuery({
    queryKey: ["auth", "sign-in-code"],
    queryFn: fetchPendingSignInCode,
    staleTime: Infinity,
    retry: false,
  });
  const pending = status.data ?? null;
  const [screen, setScreen] = useState<Screen>("code");

  return {
    // Nothing waits on a code here: it was finished, ran out, or never
    // started in this browser. A lookup that failed still lets the code be
    // typed: the code itself is checked either way.
    screen: status.isSuccess && !pending ? SIGN_IN_CODE_EXPIRED : screen,
    isLoading: status.isPending,
    email: pending?.email ?? null,
    providerName: pending ? signInProviderName(pending.provider) : null,
    // When "Send a new code" opens, from the time the server answered.
    resendAt: status.dataUpdatedAt + (pending?.resendInSeconds ?? 0) * 1000,
    handleEnded: (end: SignInCodeEnd) => setScreen(end),
    handleWaitlisted: () => setScreen("waitlisted"),
  };
}
