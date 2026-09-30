import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { ChangeSessionConnectionRequestLlmAuthProvider } from "@/app/api/__generated__/models/changeSessionConnectionRequestLlmAuthProvider";

import type { ProviderFailure } from "../../providerFailure";

export interface Alternative {
  display_name: string;
  auth_provider: ChangeSessionConnectionRequestLlmAuthProvider;
  credential_id: string | null;
}

/**
 * A connection this chat could continue on, or nothing.
 *
 * The PRD's priority order starts with the platform, because it is the one
 * connection a user is guaranteed to be able to fall back to. Anything else
 * that is selectable will do after that; the one thing excluded is the
 * connection that just refused the turn, which would fail again immediately.
 */
export function alternativeConnection(
  offers: AIConnectionOffer[] | undefined,
  failure: ProviderFailure | null,
): Alternative | null {
  if (!failure) return null;

  const usable = (offers ?? []).filter(
    (offer) =>
      offer.selectable &&
      !isTheOneThatFailed(offer, failure) &&
      (offer.auth_method === "deployment" || offer.credential_id !== null),
  );
  if (usable.length === 0) return null;

  const platform =
    usable.find((offer) => offer.auth_method === "deployment") ?? usable[0];

  return {
    display_name: platform.display_name,
    auth_provider: platform.offer_id.split(
      ":",
    )[0] as ChangeSessionConnectionRequestLlmAuthProvider,
    credential_id: platform.credential_id ?? null,
  };
}

function isTheOneThatFailed(
  offer: AIConnectionOffer,
  failure: ProviderFailure,
): boolean {
  const provider = offer.offer_id.split(":")[0];
  if (provider !== failure.authProvider) return false;
  // A provider with no credential is a single connection, so the provider
  // matching is enough. With one, only the account that failed is excluded --
  // a second ChatGPT account has its own quota.
  if (failure.credentialId === null) return true;
  return offer.credential_id === failure.credentialId;
}

/**
 * "Resets in about 3 hours", or nothing.
 *
 * Only when the provider actually reported a reset time. An invented one is
 * worse than silence, because people plan around it.
 */
export function formatResetHint(resetsAt: number | null): string | null {
  if (resetsAt === null) return null;
  const seconds = resetsAt - Math.floor(Date.now() / 1000);
  if (seconds <= 0) return null;
  if (seconds < 90) return "It resets in under a minute.";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `It resets in about ${minutes} minutes.`;
  const hours = Math.round(minutes / 60);
  return hours === 1
    ? "It resets in about an hour."
    : `It resets in about ${hours} hours.`;
}
