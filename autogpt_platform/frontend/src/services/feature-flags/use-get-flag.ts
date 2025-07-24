import { useFlags } from "launchdarkly-react-client-sdk";

export enum Flag {
  BETA_BLOCKS = "beta-blocks",
  AGENT_ACTIVITY = "agent-activity",
}

export type FlagValues = {
  [Flag.BETA_BLOCKS]: string[];
  [Flag.AGENT_ACTIVITY]: boolean;
};

const isTest = process.env.NEXT_PUBLIC_PW_TEST === "true";

const mockFlags = {
  [Flag.BETA_BLOCKS]: [],
  [Flag.AGENT_ACTIVITY]: true,
};

export function useGetFlag<T extends Flag>(flag: T): FlagValues[T] | null {
  const currentFlags = useFlags<FlagValues>();
  const flagValue = currentFlags[flag];

  if (isTest) return mockFlags[flag];
  if (!flagValue) return null;

  return flagValue;
}
