"use client";

import { DEFAULT_SEARCH_TERMS } from "@/app/(platform)/marketplace/components/HeroSection/helpers";
import { environment } from "@/services/environment";
import { useFlags } from "launchdarkly-react-client-sdk";

export enum Flag {
  BETA_BLOCKS = "beta-blocks",
  NEW_BLOCK_MENU = "new-block-menu",
  NEW_AGENT_RUNS = "new-agent-runs",
  GRAPH_SEARCH = "graph-search",
  ENABLE_ENHANCED_OUTPUT_HANDLING = "enable-enhanced-output-handling",
  NEW_FLOW_EDITOR = "new-flow-editor",
  BUILDER_VIEW_SWITCH = "builder-view-switch",
  SHARE_EXECUTION_RESULTS = "share-execution-results",
  AGENT_FAVORITING = "agent-favoriting",
  MARKETPLACE_SEARCH_TERMS = "marketplace-search-terms",
  ENABLE_PLATFORM_PAYMENT = "enable-platform-payment",
  CHAT = "chat",
}

const isPwMockEnabled = process.env.NEXT_PUBLIC_PW_TEST === "true";
const isNewFlowEditorEnvEnabled =
  process.env.NEXT_PUBLIC_NEW_FLOW_EDITOR === "true";
const isBuilderViewSwitchEnvEnabled =
  process.env.NEXT_PUBLIC_BUILDER_VIEW_SWITCH === "true";

// Local dev override: these env vars can force-enable FlowEditor flags when
// LaunchDarkly is unavailable. Defaults remain false when env vars are unset.
const defaultFlags = {
  [Flag.BETA_BLOCKS]: [],
  [Flag.NEW_BLOCK_MENU]: false,
  [Flag.NEW_AGENT_RUNS]: false,
  [Flag.GRAPH_SEARCH]: false,
  [Flag.ENABLE_ENHANCED_OUTPUT_HANDLING]: false,
  [Flag.NEW_FLOW_EDITOR]: isNewFlowEditorEnvEnabled,
  [Flag.BUILDER_VIEW_SWITCH]: isBuilderViewSwitchEnvEnabled,
  [Flag.SHARE_EXECUTION_RESULTS]: false,
  [Flag.AGENT_FAVORITING]: false,
  [Flag.MARKETPLACE_SEARCH_TERMS]: DEFAULT_SEARCH_TERMS,
  [Flag.ENABLE_PLATFORM_PAYMENT]: false,
  [Flag.CHAT]: false,
};

type FlagValues = typeof defaultFlags;

export function useGetFlag<T extends Flag>(flag: T): FlagValues[T] {
  const currentFlags = useFlags<FlagValues>();
  const flagValue = currentFlags[flag];
  const areFlagsEnabled = environment.areFeatureFlagsEnabled();

  if (!areFlagsEnabled || isPwMockEnabled) {
    return defaultFlags[flag];
  }

  return flagValue ?? defaultFlags[flag];
}
