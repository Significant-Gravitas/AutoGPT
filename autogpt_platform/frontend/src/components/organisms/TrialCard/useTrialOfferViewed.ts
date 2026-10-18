import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { TrialEvent } from "@/services/analytics/posthog-events";
import { usePostHog } from "@posthog/react";
import { useEffect, useRef } from "react";

interface Args {
  offer: TrialOfferResponse | null | undefined;
  userID: string | undefined;
  surface: "onboarding" | "billing";
}

export function useTrialOfferViewed({ offer, userID, surface }: Args) {
  const posthog = usePostHog();
  const seenOffer = useRef<string | null>(null);

  useEffect(() => {
    if (!offer || !userID) return;
    const identity = `${userID}:${offer.token}`;
    if (seenOffer.current === identity) return;
    seenOffer.current = identity;
    posthog?.capture(TrialEvent.TRIAL_OFFER_VIEWED, {
      trial_offer_version: offer.version,
      subscription_tier: offer.tier,
      trial_duration_days: offer.duration_days,
      surface,
    });
  }, [offer, userID, posthog, surface]);
}
