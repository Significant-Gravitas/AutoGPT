import { useQuery } from "@tanstack/react-query";
import Constants from "expo-constants";
import { router } from "expo-router";
import { useState } from "react";
import { Linking } from "react-native";

import { planName } from "@/features/plan/helpers";
import { usePlanCheck } from "@/features/plan/usePlanGate";
import { billingKeys, getSubscription } from "@/lib/api/billing";
import { useSession } from "@/lib/auth/nativeSession";
import { signOut } from "@/lib/session";

import { HELP_URL, PRIVACY_POLICY_URL, TERMS_OF_USE_URL } from "./helpers";

export function useSettingsScreen() {
  const session = useSession();
  const check = usePlanCheck(session?.user.id);
  // A server that sells no plans has nothing to show here.
  const isPaymentEnabled = check.data?.isPaymentEnabled === true;
  const subscription = useQuery({
    queryKey: billingKeys.subscription,
    queryFn: () => getSubscription(),
    enabled: isPaymentEnabled,
  });
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);

  function openWeb(url: string) {
    Linking.openURL(url).catch(() => undefined);
  }

  function openPlan() {
    router.navigate("/plan");
  }

  function handleSignOut() {
    void signOut();
  }

  return {
    name: session?.user.name?.trim() ?? "",
    email: session?.user.email ?? "",
    image: session?.user.image ?? null,
    isPaymentEnabled,
    planLabel: subscription.data ? planName(subscription.data.tier) : undefined,
    version: Constants.expoConfig?.version ?? null,
    isDeleteOpen,
    openDelete: () => setIsDeleteOpen(true),
    closeDelete: () => setIsDeleteOpen(false),
    openPlan,
    openHelp: () => openWeb(HELP_URL),
    openTerms: () => openWeb(TERMS_OF_USE_URL),
    openPrivacy: () => openWeb(PRIVACY_POLICY_URL),
    signOut: handleSignOut,
  };
}
