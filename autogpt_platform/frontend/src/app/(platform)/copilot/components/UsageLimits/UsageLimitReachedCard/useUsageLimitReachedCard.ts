import { useUsageActions } from "@/services/usageExperience/useUsageActions";
import type { ProviderFailure } from "../../../providerFailure";
import { useProviderLimitDialog } from "../../ProviderLimitDialog/useProviderLimitDialog";
import type { useRateLimitRefresh } from "../../RateLimitResetDialog/useRateLimitRefresh";

export interface UsageNoticeContext {
  sessionID?: string | null;
  failure?: ProviderFailure | null;
  onDismiss?: () => void;
  refresh?: Pick<
    ReturnType<typeof useRateLimitRefresh>,
    "checking" | "failed" | "refresh"
  >;
}
const dismissNothing = () => undefined;
export function useUsageLimitReachedCard({
  sessionID = null,
  failure = null,
  onDismiss = dismissNothing,
  refresh,
}: UsageNoticeContext = {}) {
  const state = useUsageActions();
  const platformCap: ProviderFailure | null =
    sessionID && state.experience.blocked && !state.isError
      ? {
          kind: "usage_limit",
          message: "Usage limit reached",
          authProvider: "platform",
          credentialId: null,
          resetsAt: null,
          retryable: false,
          reconnectFixesIt: false,
        }
      : null;
  const provider = useProviderLimitDialog({
    failure: failure ?? platformCap,
    sessionId: sessionID,
    onDismiss,
  });
  return {
    ...state,
    isLoading: state.isLoading && !refresh?.checking && !refresh?.failed,
    isError: state.isError || !!refresh?.failed || !!refresh?.checking,
    isRefreshing: !!refresh?.checking,
    retry: refresh?.refresh ?? state.retry,
    alternative: sessionID ? provider.alternative : null,
    continueHere: provider.continueHere,
    isSwitching: provider.isSwitching,
  };
}
