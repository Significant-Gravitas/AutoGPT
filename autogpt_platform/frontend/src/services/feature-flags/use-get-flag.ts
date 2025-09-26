"use client";

import { BehaveAs, getBehaveAs } from "@/lib/utils";
import { useFlags } from "launchdarkly-react-client-sdk";

export enum Flag {
  BETA_BLOCKS = "beta-blocks",
  AGENT_ACTIVITY = "agent-activity",
  NEW_BLOCK_MENU = "new-block-menu",
  NEW_AGENT_RUNS = "new-agent-runs",
  GRAPH_SEARCH = "graph-search",
  ENABLE_ENHANCED_OUTPUT_HANDLING = "enable-enhanced-output-handling",
  NEW_FLOW_EDITOR = "new-flow-editor",
  BUILDER_VIEW_SWITCH = "builder-view-switch",
  SHARE_EXECUTION_RESULTS = "share-execution-results",
  AGENT_FAVORITING = "agent-favoriting",
}

export type FlagValues = {
  [Flag.BETA_BLOCKS]: string[];
  [Flag.AGENT_ACTIVITY]: boolean;
  [Flag.NEW_BLOCK_MENU]: boolean;
  [Flag.NEW_AGENT_RUNS]: boolean;
  [Flag.GRAPH_SEARCH]: boolean;
  [Flag.ENABLE_ENHANCED_OUTPUT_HANDLING]: boolean;
  [Flag.NEW_FLOW_EDITOR]: boolean;
  [Flag.BUILDER_VIEW_SWITCH]: boolean;
  [Flag.SHARE_EXECUTION_RESULTS]: boolean;
  [Flag.AGENT_FAVORITING]: boolean;
};

const isPwMockEnabled = process.env.NEXT_PUBLIC_PW_TEST === "true";

const mockFlags = {
  [Flag.BETA_BLOCKS]: [],
  [Flag.AGENT_ACTIVITY]: true,
  [Flag.NEW_BLOCK_MENU]: false,
  [Flag.NEW_AGENT_RUNS]: false,
  [Flag.GRAPH_SEARCH]: true,
  [Flag.ENABLE_ENHANCED_OUTPUT_HANDLING]: false,
  [Flag.NEW_FLOW_EDITOR]: false,
  [Flag.BUILDER_VIEW_SWITCH]: false,
  [Flag.SHARE_EXECUTION_RESULTS]: false,
  [Flag.AGENT_FAVORITING]: false,
};

export function useGetFlag<T extends Flag>(flag: T): FlagValues[T] | null {
  const currentFlags = useFlags<FlagValues>();
  const flagValue = currentFlags[flag];
  const isCloud = getBehaveAs() === BehaveAs.CLOUD;

  if (isPwMockEnabled && !isCloud) return mockFlags[flag];

  return flagValue;
}
