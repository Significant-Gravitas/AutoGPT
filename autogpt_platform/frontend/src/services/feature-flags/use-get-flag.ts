"use client";

import { DEFAULT_SEARCH_TERMS } from "@/app/(platform)/marketplace/components/HeroSection/helpers";
import { environment } from "@/services/environment";
import { useFlags } from "launchdarkly-react-client-sdk";
import { useEffect, useState } from "react";

export enum Flag {
  BETA_BLOCKS = "beta-blocks",
  MARKETPLACE_SEARCH_TERMS = "marketplace-search-terms",
  ENABLE_PLATFORM_PAYMENT = "enable-platform-payment",
  ARTIFACTS = "artifacts",
  ARTIFACTS_PAGE = "artifacts-page",
  CHAT_MODE_OPTION = "chat-mode-option",
  BUILDER_CHAT_PANEL = "builder-chat-panel",
  AGENT_BRIEFING = "agent-briefing",
  GENERIC_TRIGGER_AGENTS = "generic-trigger-agents",
  CHAT_SEARCH = "chat-search",
  CHAT_SHARING = "chat-sharing",
  // Graphiti memory + dream-system gates. Mirror of the backend
  // ``Flag`` enum in ``backend/util/feature_flag.py``. Frontend reads
  // them when memory/dream-related UI surfaces ship (P6+ on the
  // dream-system roadmap). They default false below to match the
  // backend's fail-closed gating (default=False, opt-in only) — a
  // LaunchDarkly outage or missing flag key must not switch the
  // feature on. Use ``NEXT_PUBLIC_FORCE_FLAG_*`` env overrides to
  // enable the stack for local-dev / Playwright runs.
  GRAPHITI_MEMORY = "graphiti-memory",
  GRAPHITI_COMMUNITIES_ENABLED = "graphiti-communities-enabled",
  DREAM_PASS_ENABLED = "dream-pass-enabled",
  DREAM_PASS_WEB_FACT_CHECK = "dream-pass-web-fact-check",
  DREAM_PASS_INVALIDATE_ENTITY = "dream-pass-invalidate-entity",
}

const isPwMockEnabled = process.env.NEXT_PUBLIC_PW_TEST === "true";

const defaultFlags = {
  [Flag.BETA_BLOCKS]: [],
  [Flag.MARKETPLACE_SEARCH_TERMS]: DEFAULT_SEARCH_TERMS,
  [Flag.ENABLE_PLATFORM_PAYMENT]: false,
  [Flag.ARTIFACTS]: false,
  [Flag.ARTIFACTS_PAGE]: false,
  [Flag.CHAT_MODE_OPTION]: false,
  [Flag.BUILDER_CHAT_PANEL]: false,
  [Flag.AGENT_BRIEFING]: true,
  [Flag.GENERIC_TRIGGER_AGENTS]: false,
  [Flag.CHAT_SEARCH]: false,
  [Flag.CHAT_SHARING]: false,
  [Flag.GRAPHITI_MEMORY]: false,
  [Flag.GRAPHITI_COMMUNITIES_ENABLED]: false,
  [Flag.DREAM_PASS_ENABLED]: false,
  [Flag.DREAM_PASS_WEB_FACT_CHECK]: false,
  [Flag.DREAM_PASS_INVALIDATE_ENTITY]: false,
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
 *
 * Each flag is mapped via a literal ``process.env.NEXT_PUBLIC_FORCE_FLAG_X``
 * lookup so Next.js / Turbopack can statically inline the value into the
 * client bundle. A dynamic ``process.env[envName]`` lookup compiles to a
 * runtime read of the browser-side polyfilled ``process.env`` object,
 * which is always empty — so the override silently no-ops in dev.
 */
function readEnvOverride(flag: Flag): string | undefined {
  switch (flag) {
    case Flag.BETA_BLOCKS:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_BETA_BLOCKS;
    case Flag.MARKETPLACE_SEARCH_TERMS:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_MARKETPLACE_SEARCH_TERMS;
    case Flag.ENABLE_PLATFORM_PAYMENT:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_ENABLE_PLATFORM_PAYMENT;
    case Flag.ARTIFACTS:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_ARTIFACTS;
    case Flag.ARTIFACTS_PAGE:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_ARTIFACTS_PAGE;
    case Flag.CHAT_MODE_OPTION:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_CHAT_MODE_OPTION;
    case Flag.BUILDER_CHAT_PANEL:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_BUILDER_CHAT_PANEL;
    case Flag.AGENT_BRIEFING:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_AGENT_BRIEFING;
    case Flag.GENERIC_TRIGGER_AGENTS:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_GENERIC_TRIGGER_AGENTS;
    case Flag.CHAT_SEARCH:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_CHAT_SEARCH;
    case Flag.CHAT_SHARING:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_CHAT_SHARING;
    case Flag.GRAPHITI_MEMORY:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY;
    case Flag.GRAPHITI_COMMUNITIES_ENABLED:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_COMMUNITIES_ENABLED;
    case Flag.DREAM_PASS_ENABLED:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_DREAM_PASS_ENABLED;
    case Flag.DREAM_PASS_WEB_FACT_CHECK:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_DREAM_PASS_WEB_FACT_CHECK;
    case Flag.DREAM_PASS_INVALIDATE_ENTITY:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_DREAM_PASS_INVALIDATE_ENTITY;
  }
}

// Array-typed flags (e.g. ``BETA_BLOCKS``, ``MARKETPLACE_SEARCH_TERMS``)
// cannot be meaningfully overridden through a single boolean string env
// var — returning ``true`` / ``false`` would clash with the array type
// callers expect.  These flags are still subject to LaunchDarkly + the
// ``defaultFlags`` fallback; the env override path just skips them.
const ARRAY_TYPED_FLAGS: ReadonlySet<Flag> = new Set([
  Flag.BETA_BLOCKS,
  Flag.MARKETPLACE_SEARCH_TERMS,
]);

export function envFlagOverride<T extends Flag>(
  flag: T,
): FlagValues[T] | undefined {
  if (ARRAY_TYPED_FLAGS.has(flag)) return undefined;
  const raw = readEnvOverride(flag);
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

const FLAG_RESOLUTION_TIMEOUT_MS = 5000;

/**
 * Same as ``useGetFlag`` but also surfaces whether LaunchDarkly has
 * actually answered for this flag. Callers that gate a whole route on a
 * flag should branch on ``ready`` first — short-circuiting to
 * ``notFound()`` before LD responds 404s users that actually have the
 * flag on. Falls back to "ready" after ``FLAG_RESOLUTION_TIMEOUT_MS`` so
 * a flag key that LD never registers doesn't spin forever.
 */
export function useFlagStatus<T extends Flag>(
  flag: T,
): { enabled: FlagValues[T]; ready: boolean } {
  const currentFlags = useFlags<FlagValues>();
  const areFlagsEnabled = environment.areFeatureFlagsEnabled();
  const override = envFlagOverride(flag);

  const [timedOut, setTimedOut] = useState(false);
  useEffect(() => {
    const timer = setTimeout(
      () => setTimedOut(true),
      FLAG_RESOLUTION_TIMEOUT_MS,
    );
    return () => clearTimeout(timer);
  }, []);

  if (override !== undefined) {
    return { enabled: override, ready: true };
  }
  if (!areFlagsEnabled || isPwMockEnabled) {
    return { enabled: defaultFlags[flag], ready: true };
  }

  const ldResponded = flag in currentFlags;
  return {
    enabled: (currentFlags[flag] ?? defaultFlags[flag]) as FlagValues[T],
    ready: ldResponded || timedOut,
  };
}
