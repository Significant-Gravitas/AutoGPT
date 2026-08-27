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

export interface SubscriptionOption {
  authProvider: string;
  displayName: string;
  connectLabel: string;
  models: string;
}

/**
 * The subscriptions this deployment can offer to link, in the server's order.
 *
 * This screen used to offer exactly one, named in the copy and hardcoded in
 * the OAuth call. A deployment that enables a second would have kept sending
 * everyone to the first -- and there is no signal on screen that the other
 * exists, so nobody would report it.
 *
 * The platform route is excluded: it is not something a user links, it is
 * the API key in a file that this step is a way around.
 */
export function subscriptionOptions(
  providers: ProviderTiers[] | undefined,
): SubscriptionOption[] {
  return (providers ?? [])
    .filter(
      (provider) =>
        provider.auth_provider && provider.auth_provider !== "platform",
    )
    .map((provider) => ({
      authProvider: provider.auth_provider as string,
      displayName: provider.display_name,
      connectLabel:
        provider.connect_button_label ??
        `Sign in with ${provider.display_name}`,
      models: modelsSentence(provider),
    }));
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
function modelsSentence(provider: ProviderTiers): string {
  const named = (provider.tiers ?? [])
    .filter((tier) => tier.display_model)
    .map((tier) => `${tier.display_model} (${tier.label})`);
  if (named.length === 0) return "";
  if (named.length === 1) return named[0];
  return `${named.slice(0, -1).join(", ")} and ${named[named.length - 1]}`;
}
