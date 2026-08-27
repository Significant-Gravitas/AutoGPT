"use client";

import {
  useGetV2ListChatConnections,
  useGetV2ListProviderModelTiers,
} from "@/app/api/__generated__/endpoints/chat/chat";

import { useOnboardingWizardStore } from "../../store";
import { hasLinkedSubscription, subscriptionOptions } from "./helpers";

export function useConnectStep() {
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);

  const connectionsQuery = useGetV2ListChatConnections({
    query: { refetchOnWindowFocus: false },
  });
  const offers =
    connectionsQuery.data?.status === 200
      ? connectionsQuery.data.data.offers
      : undefined;

  // Which subscriptions this deployment offers, and what each one's tiers
  // resolve to -- neither of which the connections list can answer for a
  // connection the user has not made yet.
  const tiersQuery = useGetV2ListProviderModelTiers({
    query: { refetchOnWindowFocus: false },
  });
  const providers =
    tiersQuery.data?.status === 200
      ? tiersQuery.data.data.providers
      : undefined;

  return {
    skip: nextStep,
    isAlreadyLinked: hasLinkedSubscription(offers),
    options: subscriptionOptions(providers),
  };
}
