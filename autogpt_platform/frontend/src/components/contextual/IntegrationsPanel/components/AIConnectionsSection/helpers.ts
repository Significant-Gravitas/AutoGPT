import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { SetDefaultTransportRequest } from "@/app/api/__generated__/models/setDefaultTransportRequest";
import type { SetDefaultTransportRequestAuthProvider } from "@/app/api/__generated__/models/setDefaultTransportRequestAuthProvider";

/**
 * The route to send when making a connection the default.
 *
 * The offer describes a connection; the default is set on the transport that
 * runs it, which is keyed by provider and credential. ``offer_id`` is built
 * server-side as ``{auth_provider}:{credential_id or "deployment"}``, so the
 * provider is recoverable from it without the client deciding anything.
 */
export function routeOf(offer: AIConnectionOffer): SetDefaultTransportRequest {
  return {
    auth_provider: offer.offer_id.split(
      ":",
    )[0] as SetDefaultTransportRequestAuthProvider,
    credential_id: offer.credential_id ?? null,
  };
}

/**
 * Whether this connection can be chosen as the default.
 *
 * A locked offer is listed so the user can see it exists and what unlocks it,
 * but it cannot be routed to, so it cannot be a default either.
 */
export function isSelectable(offer: AIConnectionOffer): boolean {
  return (
    offer.selectable &&
    (offer.auth_method === "deployment" || offer.credential_id !== null)
  );
}

/**
 * Every connection worth showing, selectable or not.
 *
 * A locked one earns its place by explaining an absence the user would
 * otherwise have to guess at.
 */
export function visibleOffers(
  offers: AIConnectionOffer[] | undefined,
): AIConnectionOffer[] {
  return (offers ?? []).filter(
    (offer) => isSelectable(offer) || Boolean(offer.lock_reason),
  );
}

/**
 * "Balanced: Sonnet 5 · Advanced: Opus 5".
 *
 * Empty when the server named no models, so the row omits the line rather
 * than rendering half of one.
 */
export function tierSummary(offer: AIConnectionOffer): string {
  const named = offer.tiers
    .filter((tier) => tier.display_model)
    .map((tier) => `${tier.label}: ${tier.display_model}`);
  return named.join(" · ");
}
