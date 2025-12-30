"use client";

import { DEFAULT_SEARCH_TERMS } from "@/app/(platform)/marketplace/components/HeroSection/helpers";
import { useFlags } from "launchdarkly-react-client-sdk";
import { environment } from "../environment";

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

export type FlagValues = {
  [Flag.BETA_BLOCKS]: string[];
  [Flag.NEW_BLOCK_MENU]: boolean;
  [Flag.NEW_AGENT_RUNS]: boolean;
  [Flag.GRAPH_SEARCH]: boolean;
  [Flag.ENABLE_ENHANCED_OUTPUT_HANDLING]: boolean;
  [Flag.NEW_FLOW_EDITOR]: boolean;
  [Flag.BUILDER_VIEW_SWITCH]: boolean;
  [Flag.SHARE_EXECUTION_RESULTS]: boolean;
  [Flag.AGENT_FAVORITING]: boolean;
  [Flag.MARKETPLACE_SEARCH_TERMS]: string[];
  [Flag.ENABLE_PLATFORM_PAYMENT]: boolean;
  [Flag.CHAT]: boolean;
};

const isPwMockEnabled = process.env.NEXT_PUBLIC_PW_TEST === "true";

const mockFlags = {
  [Flag.BETA_BLOCKS]: [],
  [Flag.NEW_BLOCK_MENU]: false,
  [Flag.NEW_AGENT_RUNS]: false,
  [Flag.GRAPH_SEARCH]: true,
  [Flag.ENABLE_ENHANCED_OUTPUT_HANDLING]: false,
  [Flag.NEW_FLOW_EDITOR]: false,
  [Flag.BUILDER_VIEW_SWITCH]: false,
  [Flag.SHARE_EXECUTION_RESULTS]: false,
  [Flag.AGENT_FAVORITING]: false,
  [Flag.MARKETPLACE_SEARCH_TERMS]: DEFAULT_SEARCH_TERMS,
  [Flag.ENABLE_PLATFORM_PAYMENT]: false,
  [Flag.CHAT]: true,
};

export function useGetFlag<T extends Flag>(flag: T): FlagValues[T] | null {
  const currentFlags = useFlags<FlagValues>();
  const flagValue = currentFlags[flag];
  const isCloud = environment.isCloud();

  if ((isPwMockEnabled && !isCloud) || flagValue === undefined) {
    return mockFlags[flag];
  }

  return flagValue;
}
