import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Linking } from "react-native";

import { billingKeys } from "@/lib/api/billing";
import { useSession } from "@/lib/auth/nativeSession";
import { FRONTEND_URL } from "@/lib/env";
import { signOut } from "@/lib/session";

const STILL_NO_PLAN = "Still no active plan on this account.";
const OFFLINE = "Can't reach AutoGPT. Check your connection and try again.";
const NO_BROWSER = "Couldn't open your browser. Try again.";

export function useNoPlanScreen() {
  const session = useSession();
  const queryClient = useQueryClient();
  const [isChecking, setIsChecking] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  // The root layout mounts this screen afresh for each account.
  const checkKey = billingKeys.check(session?.user.id ?? "");

  async function openSignup() {
    setNotice(null);
    try {
      await Linking.openURL(`${FRONTEND_URL}/signup`);
    } catch {
      setNotice(NO_BROWSER);
    }
  }

  // A plan swaps this screen for Home; anything said here is for no plan.
  async function checkAgain() {
    setIsChecking(true);
    setNotice(null);
    await queryClient.refetchQueries({ queryKey: checkKey });
    const failed = queryClient.getQueryState(checkKey)?.status === "error";
    setNotice(failed ? OFFLINE : STILL_NO_PLAN);
    setIsChecking(false);
  }

  function handleSignOut() {
    void signOut();
  }

  return {
    email: session?.user.email ?? "",
    name: session?.user.name ?? "",
    image: session?.user.image ?? null,
    isChecking,
    notice,
    openSignup,
    checkAgain,
    signOut: handleSignOut,
  };
}
