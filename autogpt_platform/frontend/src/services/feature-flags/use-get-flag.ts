"use client";

import { DEFAULT_SEARCH_TERMS } from "@/app/(platform)/marketplace/components/HeroSection/helpers";
import { environment } from "@/services/environment";
import { useFlags } from "launchdarkly-react-client-sdk";

export enum Flag {
  BETA_BLOCKS = "beta-blocks",
  MARKETPLACE_SEARCH_TERMS = "marketplace-search-terms",
  ENABLE_PLATFORM_PAYMENT = "enable-platform-payment",
  ARTIFACTS = "artifacts",
  CHAT_MODE_OPTION = "chat-mode-option",
  BUILDER_CHAT_PANEL = "builder-chat-panel",
  AGENT_BRIEFING = "agent-briefing",
  GENERIC_TRIGGER_AGENTS = "generic-trigger-agents",
  CHAT_SEARCH = "chat-search",
  COPILOT_SKILLS_FOLLOWUPS = "copilot-skills-followups",
}

const isPwMockEnabled = process.env.NEXT_PUBLIC_PW_TEST === "true";

const defaultFlags = {
  [Flag.BETA_BLOCKS]: [],
  [Flag.MARKETPLACE_SEARCH_TERMS]: DEFAULT_SEARCH_TERMS,
  [Flag.ENABLE_PLATFORM_PAYMENT]: false,
  [Flag.ARTIFACTS]: false,
  [Flag.CHAT_MODE_OPTION]: false,
  [Flag.BUILDER_CHAT_PANEL]: false,
  [Flag.AGENT_BRIEFING]: false,
  [Flag.GENERIC_TRIGGER_AGENTS]: false,
  [Flag.CHAT_SEARCH]: false,
  [Flag.COPILOT_SKILLS_FOLLOWUPS]: false,
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
    case Flag.COPILOT_SKILLS_FOLLOWUPS:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_COPILOT_SKILLS_FOLLOWUPS;
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
