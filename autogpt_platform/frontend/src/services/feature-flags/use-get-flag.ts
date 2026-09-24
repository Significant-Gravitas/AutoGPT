"use client";

import { DEFAULT_SEARCH_TERMS } from "@/app/(platform)/marketplace/components/HeroSection/helpers";
import { environment } from "@/services/environment";
import * as Sentry from "@sentry/nextjs";
import type { FeatureFlagsIntegration } from "@sentry/nextjs";
import { useEffect, useState } from "react";
import { FLAG_BACKEND, isPostHogFlagsEnabled } from "./flag-backend";
import { useFlagSource } from "./flag-source";

export enum Flag {
  MARKETPLACE_SEARCH_TERMS = "marketplace-search-terms",
  ENABLE_PLATFORM_PAYMENT = "enable-platform-payment",
  ARTIFACTS_PAGE = "artifacts-page",
  CHAT_MODE_OPTION = "chat-mode-option",
  GENERIC_TRIGGER_AGENTS = "generic-trigger-agents",
  CHAT_SEARCH = "chat-search",
  AUTOGPT_NEW_LAYOUT = "autogpt-new-layout",
  CHAT_WORKSPACE_FILES = "chat-workspace-files",
  CHAT_PINNING = "chat-pinning",
  TASK_PROGRESS_BAR = "task-progress-bar",
  HIRE_EXPERTS = "hire-experts",
  // Reveals the marketplace Skills shelf and the skill listing pages.
  SKILLS_HUB = "skills-hub",
  // Reveals the notification-preferences card on /settings/account. The card
  // is built but its design is still being reworked, so it ships dark and is
  // targeted at AGPT staff in LaunchDarkly. Until this is on for everyone,
  // /profile/settings stays un-redirected (``useNewSettingsRedirect``) so the
  // toggles remain reachable for everyone else — flip both together.
  SETTINGS_NOTIFICATIONS = "settings-notifications",
  // Replaces the onboarding pillbox step with the voice brain dump.
  // Mirror of the backend ``Flag`` enum — the endpoints 404 when off, so
  // both sides must agree. Off renders the pillbox flow untouched.
  ONBOARDING_BRAIN_DUMP = "onboarding-brain-dump",
  // Child of HIRE_EXPERTS: the greeting page builds a team from the brain
  // dump. Mirror of the backend ``Flag`` enum; both must be on.
  ONBOARDING_EXPERT_TEAM = "onboarding-expert-team",
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
  // JSON flag mapping copilot-bot platform key (lowercase) -> visible on the
  // Bots settings page. Lets ops hide a platform (e.g. Slack while its
  // Marketplace review is pending) without a deploy. Missing keys default to
  // visible — only an explicit ``false`` hides a card.
  COPILOT_BOT_PLATFORMS = "copilot-bot-platforms",
  // Voice mode on /copilot: hands-free listen → send → speak → listen.
  // Mirror of the backend ``Flag`` enum — the speech endpoint 404s when off,
  // so both sides must agree. Fail-closed.
  COPILOT_VOICE_MODE = "copilot-voice-mode",
}

const isPwMockEnabled = process.env.NEXT_PUBLIC_PW_TEST === "true";

const defaultFlags = {
  [Flag.MARKETPLACE_SEARCH_TERMS]: DEFAULT_SEARCH_TERMS,
  [Flag.ENABLE_PLATFORM_PAYMENT]: false,
  [Flag.ARTIFACTS_PAGE]: false,
  [Flag.CHAT_MODE_OPTION]: false,
  [Flag.GENERIC_TRIGGER_AGENTS]: false,
  [Flag.CHAT_SEARCH]: false,
  [Flag.AUTOGPT_NEW_LAYOUT]: false,
  [Flag.CHAT_WORKSPACE_FILES]: false,
  [Flag.CHAT_PINNING]: false,
  [Flag.TASK_PROGRESS_BAR]: false,
  [Flag.HIRE_EXPERTS]: false,
  [Flag.SKILLS_HUB]: false,
  // Off by default so a LaunchDarkly outage or a missing key hides the card
  // rather than exposing the in-progress design to everyone.
  [Flag.SETTINGS_NOTIFICATIONS]: false,
  // Off by default: with no LaunchDarkly key (local dev, CI, Playwright)
  // the wizard falls back to this map, and a ``true`` here renders the
  // brain dump for everyone — which is what the backend 404s are meant to
  // prevent. Use NEXT_PUBLIC_FORCE_FLAG_ONBOARDING_BRAIN_DUMP locally.
  [Flag.ONBOARDING_BRAIN_DUMP]: false,
  [Flag.ONBOARDING_EXPERT_TEAM]: false,
  [Flag.GRAPHITI_MEMORY]: false,
  [Flag.GRAPHITI_COMMUNITIES_ENABLED]: false,
  [Flag.DREAM_PASS_ENABLED]: false,
  [Flag.DREAM_PASS_WEB_FACT_CHECK]: false,
  [Flag.DREAM_PASS_INVALIDATE_ENTITY]: false,
  [Flag.COPILOT_BOT_PLATFORMS]: {} as Record<string, boolean>,
  [Flag.COPILOT_VOICE_MODE]: false,
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
    case Flag.MARKETPLACE_SEARCH_TERMS:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_MARKETPLACE_SEARCH_TERMS;
    case Flag.ENABLE_PLATFORM_PAYMENT:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_ENABLE_PLATFORM_PAYMENT;
    case Flag.SKILLS_HUB:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_SKILLS_HUB;
    case Flag.ARTIFACTS_PAGE:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_ARTIFACTS_PAGE;
    case Flag.CHAT_MODE_OPTION:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_CHAT_MODE_OPTION;
    case Flag.GENERIC_TRIGGER_AGENTS:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_GENERIC_TRIGGER_AGENTS;
    case Flag.CHAT_SEARCH:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_CHAT_SEARCH;
    case Flag.AUTOGPT_NEW_LAYOUT:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_AUTOGPT_NEW_LAYOUT;
    case Flag.CHAT_WORKSPACE_FILES:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_CHAT_WORKSPACE_FILES;
    case Flag.CHAT_PINNING:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_CHAT_PINNING;
    case Flag.TASK_PROGRESS_BAR:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_TASK_PROGRESS_BAR;
    case Flag.HIRE_EXPERTS:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_HIRE_EXPERTS;
    case Flag.SETTINGS_NOTIFICATIONS:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_SETTINGS_NOTIFICATIONS;
    case Flag.ONBOARDING_BRAIN_DUMP:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_ONBOARDING_BRAIN_DUMP;
    case Flag.ONBOARDING_EXPERT_TEAM:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_ONBOARDING_EXPERT_TEAM;
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
    case Flag.COPILOT_VOICE_MODE:
      return process.env.NEXT_PUBLIC_FORCE_FLAG_COPILOT_VOICE_MODE;
    case Flag.COPILOT_BOT_PLATFORMS:
      return undefined;
  }
}

// Array-typed flags (e.g. ``MARKETPLACE_SEARCH_TERMS``)
// cannot be meaningfully overridden through a single boolean string env
// var — returning ``true`` / ``false`` would clash with the array type
// callers expect.  These flags are still subject to LaunchDarkly + the
// ``defaultFlags`` fallback; the env override path just skips them.
const ARRAY_TYPED_FLAGS: ReadonlySet<Flag> = new Set([
  Flag.MARKETPLACE_SEARCH_TERMS,
  Flag.COPILOT_BOT_PLATFORMS,
]);

// Master local-dev switch: ``NEXT_PUBLIC_FORCE_ALL_FLAGS=true`` turns every
// boolean flag on without listing them individually. A per-flag
// ``NEXT_PUBLIC_FORCE_FLAG_<NAME>`` still wins, so one flag can be excluded
// with ``=false`` while the rest stay forced. Array/JSON-typed flags keep
// their LaunchDarkly / default values.
//
// Inert in a production build, mirroring the backend's app_env guard:
// NEXT_PUBLIC_* vars are inlined at build time, so one stray value in a
// production env file would otherwise bake every fail-closed gate open into
// the client bundle. Per-flag overrides are unaffected.
const isForceAllFlags =
  process.env.NODE_ENV !== "production" &&
  ["1", "true", "yes", "on"].includes(
    (process.env.NEXT_PUBLIC_FORCE_ALL_FLAGS ?? "").trim().toLowerCase(),
  );

export function envFlagOverride<T extends Flag>(
  flag: T,
): FlagValues[T] | undefined {
  if (ARRAY_TYPED_FLAGS.has(flag)) return undefined;
  const raw = readEnvOverride(flag);
  if (raw === undefined) {
    return isForceAllFlags ? (true as FlagValues[T]) : undefined;
  }
  const normalized = raw.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true as FlagValues[T];
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false as FlagValues[T];
  }
  return isForceAllFlags ? (true as FlagValues[T]) : undefined;
}

export function useGetFlag<T extends Flag>(flag: T): FlagValues[T] {
  const { value } = useFlagSource(flag);
  const override = envFlagOverride(flag);
  const served = override ?? servedFlagValue(flag, value);
  recordFlagForSentry(flag, override === undefined ? served : undefined);
  return served;
}

const FLAG_RESOLUTION_TIMEOUT_MS = 5000;

/**
 * Same as ``useGetFlag`` but also surfaces whether the flag vendor has
 * actually answered for this flag. Callers that gate a whole route on a
 * flag should branch on ``ready`` first — short-circuiting to
 * ``notFound()`` before the vendor responds 404s users that actually have
 * the flag on. Falls back to "ready" after ``FLAG_RESOLUTION_TIMEOUT_MS``
 * so an unregistered flag key doesn't spin forever; ``answered`` stays
 * false then, for callers that must not act on a timeout.
 */
export function useFlagStatus<T extends Flag>(
  flag: T,
): { enabled: FlagValues[T]; ready: boolean; answered: boolean } {
  const { value, resolved } = useFlagSource(flag);
  const areFlagsEnabled = areFeatureFlagsEnabled();
  const override = envFlagOverride(flag);

  const [timedOut, setTimedOut] = useState(false);
  useEffect(() => {
    const timer = setTimeout(
      () => setTimedOut(true),
      FLAG_RESOLUTION_TIMEOUT_MS,
    );
    return () => clearTimeout(timer);
  }, []);

  const served = override ?? servedFlagValue(flag, value);
  recordFlagForSentry(flag, override === undefined ? served : undefined);

  if (override !== undefined || !areFlagsEnabled || isPwMockEnabled) {
    return { enabled: served, ready: true, answered: true };
  }
  return {
    enabled: served,
    ready: resolved || timedOut,
    answered: resolved,
  };
}

function servedFlagValue<T extends Flag>(flag: T, value: unknown) {
  if (!areFeatureFlagsEnabled() || isPwMockEnabled) return defaultFlags[flag];
  return resolveFlagValue(flag, value);
}

// PostHog answers a flag with no payload as a bare boolean, so a JSON-valued
// flag can arrive as `true` and reach a consumer that calls `.map` on it.
// `typeof` alone can't separate an array from an object; both are "object".
export function resolveFlagValue<T extends Flag>(
  flag: T,
  value: unknown,
): FlagValues[T] {
  const fallback = defaultFlags[flag];

  if (value === undefined || value === null) return fallback;

  if (Array.isArray(fallback)) {
    return (Array.isArray(value) ? value : fallback) as FlagValues[T];
  }

  if (fallback !== null && typeof fallback === "object") {
    const isPlainObject = typeof value === "object" && !Array.isArray(value);
    return (isPlainObject ? value : fallback) as FlagValues[T];
  }

  return (typeof value === typeof fallback ? value : fallback) as FlagValues[T];
}

// ``environment.areFeatureFlagsEnabled`` only knows about LaunchDarkly, and
// deliberately stays that way — it is what the provider and the flag test
// mocks stub. This is the same question asked of whichever vendor is configured.
function areFeatureFlagsEnabled() {
  switch (FLAG_BACKEND) {
    case "posthog":
      return isPostHogFlagsEnabled();
    case "dual":
      return environment.areFeatureFlagsEnabled() || isPostHogFlagsEnabled();
    default:
      return environment.areFeatureFlagsEnabled();
  }
}

// Records the value served, not the vendor's; env overrides pass undefined.
// Called during render, not in an effect: a component that throws in the same
// render never commits, and its error would reach Sentry without the flag.
// Sentry's flag context holds booleans only; JSON-valued flags are skipped.
// A recording failure must never break a flag read.
function recordFlagForSentry(key: string, value: unknown) {
  if (typeof value !== "boolean") return;
  try {
    Sentry.getClient()
      ?.getIntegrationByName<FeatureFlagsIntegration>("FeatureFlags")
      ?.addFeatureFlag(key, value);
  } catch (error) {
    // Debug, not warn: captureConsoleIntegration would send it to Sentry.
    console.debug(`Could not record flag ${key} for Sentry`, error);
  }
}
