"use client";

import { useEffect } from "react";

import { useGetV2ListChatConnections } from "@/app/api/__generated__/endpoints/chat/chat";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";

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
  // Only ever consider what can actually run: a locked offer is listed to
  // explain itself, never to be landed on.
  const choosable = offers.filter(isSelectable);
  const selected = choosable.find((offer) =>
    matchesSelection(offer, copilotLlmAuth),
  );

  // Nothing chosen yet, or the chosen connection is gone: fall in behind the
  // one the server marks default. Deciding where a chat starts is the
  // server's job — this only reflects it.
  const fallback = choosable.find((offer) => offer.is_default) ?? choosable[0];
  useEffect(() => {
    if (selected || !fallback) return;
    const selection = offerToSelection(fallback);
    if (selection) setCopilotLlmAuth(selection);
  }, [selected, fallback?.offer_id]);

  const active = selected ?? fallback;

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
    isLoading: query.isLoading,
    isError: query.isError,
  };
}
