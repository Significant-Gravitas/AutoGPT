import { useFlags } from "launchdarkly-react-client-sdk";

export function useFeatureFlag(flagKey: string): boolean {
  const flags = useFlags();
  if (flags && flagKey in flags) {
    return Boolean((flags as Record<string, boolean>)[flagKey]);
  }
  return false;
}
