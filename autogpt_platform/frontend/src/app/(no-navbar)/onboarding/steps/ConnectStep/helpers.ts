import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { ProviderTiers } from "@/app/api/__generated__/models/providerTiers";

/**
 * Whether the user has already linked a subscription of their own.
 *
 * The deployment's own chat provider does not count. It is a connection, but
 * it is the one that exists because someone put an API key in a file — which
 * is the thing this step offers a way around.
 */
export function hasLinkedSubscription(
  offers: AIConnectionOffer[] | undefined,
): boolean {
  return (offers ?? []).some(
    (offer) =>
      offer.auth_method !== "deployment" && Boolean(offer.credential_id),
  );
}

/**
 * "5.6 Terra (Balanced) and 5.6 Sol (Advanced)", from the catalog.
 *
 * Read from the provider-tiers endpoint rather than from the user's
 * connections, because the whole point of this screen is that the user does
 * not have this connection yet -- so it is absent from the offers list, and
 * the sentence would always come out empty.
 *
 * Empty when the server named nothing, so the sentence that would have
 * carried it is dropped rather than rendered half-written. Never hardcoded
 * here: which model a tier resolves to is the server's to say, and this
 * screen is the one making the promise.
 */
export function linkedModelsSentence(
  providers: ProviderTiers[] | undefined,
): string {
  const chatgpt = (providers ?? []).find(
    (provider) => provider.provider_family === "openai",
  );
  const named = (chatgpt?.tiers ?? [])
    .filter((tier) => tier.display_model)
    .map((tier) => `${tier.display_model} (${tier.label})`);
  if (named.length === 0) return "";
  if (named.length === 1) return named[0];
  return `${named.slice(0, -1).join(", ")} and ${named[named.length - 1]}`;
}
