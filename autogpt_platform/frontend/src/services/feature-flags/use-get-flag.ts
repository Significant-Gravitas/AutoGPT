import { useFlags } from "launchdarkly-react-client-sdk";

export enum Flag {
  BETA_BLOCKS = "beta-blocks",
}

export type FlagValues = {
  [Flag.BETA_BLOCKS]: string[];
};

export function useGetFlag(flag: Flag) {
  const currentFlags = useFlags<FlagValues>();
  const flagValue = currentFlags[flag];
  if (!flagValue) return null;
  return flagValue;
}
