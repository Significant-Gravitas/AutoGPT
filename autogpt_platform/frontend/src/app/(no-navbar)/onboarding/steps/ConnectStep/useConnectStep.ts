"use client";

import {
  getGetV2ListChatConnectionsQueryKey,
  useGetV2ListChatConnections,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useQueryClient } from "@tanstack/react-query";
import { useOAuthConnect } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect";

import { useOnboardingWizardStore } from "../../store";

export function useConnectStep() {
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const queryClient = useQueryClient();

  const connectionsQuery = useGetV2ListChatConnections({
    query: { refetchOnWindowFocus: false },
  });
  const offers =
    connectionsQuery.data?.status === 200
      ? connectionsQuery.data.data.offers
      : undefined;

  function finishConnection() {
    void queryClient.invalidateQueries({
      queryKey: getGetV2ListChatConnectionsQueryKey(),
    });
  }

  const { connect, isPending } = useOAuthConnect({
    provider: "codex",
    onSuccess: finishConnection,
  });

  return {
    connect,
    finishConnection,
    isConnecting: isPending,
    skip: nextStep,
    isChatGPTLinked: (offers ?? []).some(
      (offer) =>
        offer.auth_provider === "codex" && Boolean(offer.credential_id),
    ),
    isMicrosoftLinked: (offers ?? []).some(
      (offer) =>
        offer.auth_provider === "microsoft_365_copilot" &&
        Boolean(offer.credential_id),
    ),
  };
}
