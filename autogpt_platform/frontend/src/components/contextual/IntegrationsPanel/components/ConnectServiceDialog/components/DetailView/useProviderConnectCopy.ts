"use client";

import { useGetV2ListProviderModelTiers } from "@/app/api/__generated__/endpoints/chat/chat";

import { subscriptionModelsSentence } from "./helpers";

/**
 * How this dialog should describe the account it is about to link.
 *
 * The panel used to answer this itself, with a branch: "is this codex? then
 * say Sign in with ChatGPT, name OpenAI in the terms line, and show the
 * explainer". All three are the provider's own copy, the server already
 * holds them, and a second subscription provider would need the branch
 * edited rather than a row added -- so the panel asks instead.
 *
 * `isSubscription` is the same question the branch was really asking: does
 * linking this change who pays for a run? True only for a provider the
 * server describes as a chat connection, which is what makes the explainer
 * appear for ChatGPT and not for Notion.
 */
export function useProviderConnectCopy(authProvider: string | undefined) {
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
    buttonLabel: match?.connect_button_label ?? null,
    termsCompany: match?.terms_company ?? null,
    modelsSentence: subscriptionModelsSentence(providers, authProvider),
  };
}
