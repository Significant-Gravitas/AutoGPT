"use client";

import { useGetV2ListProviderModelTiers } from "@/app/api/__generated__/endpoints/chat/chat";

/**
 * How long to wait on a sign-in window for this provider, in milliseconds.
 *
 * `undefined` means "the popup helper's own default", which is right for an
 * ordinary OAuth redirect: the user approves at the provider and comes
 * straight back.
 *
 * A subscription sign-in is a longer errand -- a third party's login, an
 * account picker, and for the device-code providers a CLI polling for the
 * grant -- and the default cuts it off partway through. Both call sites used
 * to spell that out as `provider === "codex" ? 15 * 60 * 1000 : undefined`,
 * which is a claim about the sign-in strategy written as a claim about one
 * provider's name; the second provider would have silently inherited the
 * short window and timed out on a flow that was still going fine.
 *
 * The server knows which strategy each provider uses, so it sends the number.
 */
export function useOAuthLoginTimeout(provider: string | undefined) {
  const { data } = useGetV2ListProviderModelTiers({
    query: { refetchOnWindowFocus: false, enabled: Boolean(provider) },
  });

  if (!provider || data?.status !== 200) return undefined;
  const match = data.data.providers.find(
    (entry) => entry.auth_provider === provider,
  );
  const seconds = match?.login_timeout_seconds;
  return seconds ? seconds * 1000 : undefined;
}
