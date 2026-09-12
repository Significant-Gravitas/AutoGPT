"use client";

import { useGetV2ListChatConnections } from "@/app/api/__generated__/endpoints/chat/chat";
import { useGetV1ListProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { useOAuthConnect } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect";

import { useCopilotUIStore } from "../../../../store";
import {
  isSelectable,
  matchesSelection,
  offerToSelection,
  tiersAreDistinct,
  visibleOffers,
} from "./helpers";

export function useConnectionPicker() {
  const query = useGetV2ListChatConnections({
    query: { refetchOnWindowFocus: true, staleTime: 0 },
  });
  const {
    copilotLlmAuth,
    setCopilotLlmAuth,
    copilotLlmModel,
    setCopilotLlmModel,
  } = useCopilotUIStore();

  const offers = visibleOffers(
    query.data?.status === 200 ? query.data.data.offers : undefined,
  );
  const needsChatGPTConnection =
    query.data?.status === 200 &&
    offers.length > 0 &&
    offers.every((offer) => offer.provider_family !== "openai");
  const providersQuery = useGetV1ListProviders({
    query: {
      enabled: needsChatGPTConnection,
      refetchOnWindowFocus: true,
      staleTime: 0,
    },
  });
  // Only ever consider what can actually run: a locked offer is listed to
  // explain itself, never to be landed on.
  const choosable = offers.filter(isSelectable);
  const selected = choosable.find((offer) =>
    matchesSelection(offer, copilotLlmAuth),
  );

  // Nothing chosen yet, or the chosen connection is gone: show the one the
  // server marks default. Showing it is all this does -- writing it into the
  // store would turn "follow the server" into a standing choice, and the
  // store has no way back to null: a connection picked for one chat would
  // silently become the default for every later chat, a new default set in
  // Settings could never take over, and the create call would always name a
  // route, which makes the server skip its own default. Deciding where a chat
  // starts is the server's job, so an unmade choice stays unmade.
  const fallback = choosable.find((offer) => offer.is_default) ?? choosable[0];

  const active = selected ?? fallback;

  const { connect: connectChatGPT, isPending: isConnecting } = useOAuthConnect({
    provider: "codex",
    onSuccess: () => {
      query.refetch();
    },
  });

  function chooseConnection(offer: AIConnectionOffer) {
    if (!isSelectable(offer)) return;
    const selection = offerToSelection(offer);
    if (selection) setCopilotLlmAuth(selection);
  }

  return {
    offers,
    active,
    chooseConnection,
    tier: copilotLlmModel,
    setTier: setCopilotLlmModel,
    showTiers: tiersAreDistinct(active),
    // Whether there is a connection to choose *between*. Counts only what can
    // actually run: a locked offer is listed to explain an absence, so it adds
    // a row to the popover without adding a decision.
    hasConnectionChoice: choosable.length > 1,
    connectChatGPT,
    isConnecting,
    // The upsell flag can hide a locked offer. Only provider discovery can
    // confirm that an unconnected account is eligible to connect.
    canConnectChatGPT:
      needsChatGPTConnection &&
      providersQuery.isSuccess &&
      providersQuery.data.status === 200 &&
      providersQuery.data.data.some((provider) => provider.name === "codex"),
    isLoading: query.isLoading,
    // React Query can retain the last successful offers while a background
    // refetch fails. Keep that usable snapshot visible; the error state is
    // only terminal when there is nothing to render.
    isError: query.isError && offers.length === 0,
  };
}
