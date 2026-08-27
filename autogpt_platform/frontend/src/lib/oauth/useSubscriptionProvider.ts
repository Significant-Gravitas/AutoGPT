"use client";

import { useGetV2ListProviderModelTiers } from "@/app/api/__generated__/endpoints/chat/chat";
import type { ProviderTiers } from "@/app/api/__generated__/models/providerTiers";

/**
 * What the server says about a chat subscription provider, for whichever
 * surface is about to describe one.
 *
 * Four places used to answer these questions themselves, each with its own
 * `provider === "codex"` branch: the connect dialog (button label, terms
 * company, whether to explain at all), the credentials input (modal title
 * and action-button text), and both OAuth call sites (how long to wait on
 * the sign-in window). Every one of them would have been wrong for a second
 * provider, and wrong quietly -- a generic label, a missing explanation, a
 * sign-in cut off mid-flow.
 *
 * They ask here instead, and this asks the server. One query, cached across
 * all of them.
 *
 * `isSubscription` is the question those branches were really asking: does
 * linking this change who pays for a run? True only for a provider the
 * server describes as a chat connection -- so ChatGPT, and not Notion.
 */
export function useSubscriptionProvider(authProvider: string | undefined) {
  const { data } = useGetV2ListProviderModelTiers({
    query: { refetchOnWindowFocus: false, enabled: Boolean(authProvider) },
  });

  const providers = data?.status === 200 ? data.data.providers : undefined;
  const match =
    authProvider && authProvider !== "platform"
      ? providers?.find((provider) => provider.auth_provider === authProvider)
      : undefined;

  return {
    isSubscription: Boolean(match),
    displayName: match?.display_name ?? null,
    connectButtonLabel: match?.connect_button_label ?? null,
    termsCompany: match?.terms_company ?? null,
    modelsSentence: modelsSentenceFor(match),
    /**
     * `undefined` means "use the caller's own default", which is right for
     * an ordinary OAuth redirect: the user approves and comes straight back.
     */
    loginTimeoutMs: match?.login_timeout_seconds
      ? match.login_timeout_seconds * 1000
      : undefined,
  };
}

/**
 * "5.6 Terra (Balanced) and 5.6 Sol (Advanced)", from the catalog.
 *
 * Empty when the server named nothing, so a caller falls back to its general
 * sentence rather than rendering half of a promise.
 */
function modelsSentenceFor(provider: ProviderTiers | undefined): string {
  const named = (provider?.tiers ?? [])
    .filter((tier) => tier.display_model)
    .map((tier) => `${tier.display_model} (${tier.label})`);
  if (named.length === 0) return "";
  if (named.length === 1) return named[0];
  return `${named.slice(0, -1).join(", ")} and ${named[named.length - 1]}`;
}
