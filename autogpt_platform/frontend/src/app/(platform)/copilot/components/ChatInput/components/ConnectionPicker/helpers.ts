import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { CopilotLlmAuthSelection } from "../../../../store";

export function offerToSelection(
  offer: AIConnectionOffer,
): CopilotLlmAuthSelection | null {
  if (offer.auth_method === "deployment") {
    return { authProvider: "platform", credentialId: null };
  }
  if (!offer.credential_id) return null;
  return { authProvider: "codex", credentialId: offer.credential_id };
}

export function matchesSelection(
  offer: AIConnectionOffer,
  selection: CopilotLlmAuthSelection | null,
): boolean {
  if (!selection) return false;
  if (selection.authProvider === "platform") {
    return offer.auth_method === "deployment";
  }
  return offer.credential_id === selection.credentialId;
}

export function selectableOffers(
  offers: AIConnectionOffer[] | undefined,
): AIConnectionOffer[] {
  return (offers ?? []).filter(
    (offer) =>
      offer.selectable &&
      (offer.auth_method === "deployment" || offer.credential_id !== null),
  );
}

/**
 * Whether picking a tier on this connection means anything.
 *
 * A deployment can resolve both tiers to the same model — a single-model
 * self-host does exactly that — and offering a choice between two identical
 * options is a decision with no consequence. Where the server cannot name
 * the models at all, the tiers are kept: unknown is not the same as equal.
 */
export function tiersAreDistinct(
  offer: AIConnectionOffer | undefined,
): boolean {
  if (!offer || offer.tiers.length < 2) return false;
  const named = offer.tiers.filter((tier) => tier.display_model);
  if (named.length < offer.tiers.length) return true;
  return new Set(named.map((tier) => tier.display_model)).size > 1;
}

function tierOf(offer: AIConnectionOffer | undefined, tier: string) {
  return offer?.tiers.find((candidate) => candidate.tier === tier);
}

export function tierName(
  offer: AIConnectionOffer | undefined,
  tier: string,
): string {
  return (
    tierOf(offer, tier)?.label ??
    (tier === "advanced" ? "Advanced" : "Balanced")
  );
}

export function tierModel(
  offer: AIConnectionOffer | undefined,
  tier: string,
): string | null {
  return tierOf(offer, tier)?.display_model ?? null;
}

/** The whole choice in one string, for an accessible name. */
export function tierLabel(
  offer: AIConnectionOffer | undefined,
  tier: string,
): string {
  const model = tierModel(offer, tier);
  const name = tierName(offer, tier);
  return model ? `${name} · ${model}` : name;
}
