"use client";

import { DEFAULT_SEARCH_TERMS } from "@/app/(platform)/marketplace/components/HeroSection/helpers";
import { environment } from "@/services/environment";
import { useFlags } from "launchdarkly-react-client-sdk";

export enum Flag {
  BETA_BLOCKS = "beta-blocks",
  NEW_BLOCK_MENU = "new-block-menu",
  GRAPH_SEARCH = "graph-search",
  ENABLE_ENHANCED_OUTPUT_HANDLING = "enable-enhanced-output-handling",
  SHARE_EXECUTION_RESULTS = "share-execution-results",
  AGENT_FAVORITING = "agent-favoriting",
  MARKETPLACE_SEARCH_TERMS = "marketplace-search-terms",
  ENABLE_PLATFORM_PAYMENT = "enable-platform-payment",
  ARTIFACTS = "artifacts",
  CHAT_MODE_OPTION = "chat-mode-option",
}

const isPwMockEnabled = process.env.NEXT_PUBLIC_PW_TEST === "true";

const defaultFlags = {
  [Flag.BETA_BLOCKS]: [],
  [Flag.NEW_BLOCK_MENU]: false,
  [Flag.GRAPH_SEARCH]: false,
  [Flag.ENABLE_ENHANCED_OUTPUT_HANDLING]: false,
  [Flag.SHARE_EXECUTION_RESULTS]: false,
  [Flag.AGENT_FAVORITING]: false,
  [Flag.MARKETPLACE_SEARCH_TERMS]: DEFAULT_SEARCH_TERMS,
  [Flag.ENABLE_PLATFORM_PAYMENT]: false,
  [Flag.ARTIFACTS]: false,
  [Flag.CHAT_MODE_OPTION]: false,
};

type FlagValues = typeof defaultFlags;

/**
 * Read a per-flag override from the build-time env.
 *
 * Set ``NEXT_PUBLIC_FORCE_FLAG_<NAME>=true|false`` (``NAME`` = flag value
 * with ``-`` → ``_``, upper-cased) to bypass LaunchDarkly for that flag
 * in local dev.  Returns ``undefined`` when no override is configured so
 * the caller falls through to LaunchDarkly / ``defaultFlags``.
 *
 * Note: ``NEXT_PUBLIC_*`` env vars are baked into the bundle at build
 * time, so the frontend image must be rebuilt after changing them.
 */
export function envFlagOverride<T extends Flag>(
  flag: T,
): FlagValues[T] | undefined {
  const envName =
    "NEXT_PUBLIC_FORCE_FLAG_" + flag.toUpperCase().replace(/-/g, "_");
  const raw = process.env[envName];
  if (raw === undefined) return undefined;
  const normalized = raw.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true as FlagValues[T];
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false as FlagValues[T];
  }
  return undefined;
}

export function useGetFlag<T extends Flag>(flag: T): FlagValues[T] {
  const currentFlags = useFlags<FlagValues>();
  const flagValue = currentFlags[flag];
  const areFlagsEnabled = environment.areFeatureFlagsEnabled();

  const override = envFlagOverride(flag);
  if (override !== undefined) {
    return override;
  }

  if (!areFlagsEnabled || isPwMockEnabled) {
    return defaultFlags[flag];
  }

  return flagValue ?? defaultFlags[flag];
}
