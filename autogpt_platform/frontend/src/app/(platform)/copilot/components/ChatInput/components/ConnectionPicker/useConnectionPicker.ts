"use client";

import { useGetV2ListChatConnections } from "@/app/api/__generated__/endpoints/chat/chat";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";

import { useCopilotUIStore } from "../../../../store";
import {
  matchesSelection,
  offerToSelection,
  selectableOffers,
  tiersAreDistinct,
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

  const offers = selectableOffers(
    query.data?.status === 200 ? query.data.data.offers : undefined,
  );
  const selected = offers.find((offer) =>
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
  const fallback = offers.find((offer) => offer.is_default) ?? offers[0];

  const active = selected ?? fallback;

  function chooseConnection(offer: AIConnectionOffer) {
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
    isLoading: query.isLoading,
    isError: query.isError,
  };
}
