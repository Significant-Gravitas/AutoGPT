"use client";

import { useRateLimitRefresh } from "./useRateLimitRefresh";
import type { ProviderFailure } from "../../providerFailure";
import { useProviderLimitDialog } from "../ProviderLimitDialog/useProviderLimitDialog";
import { RateLimitResetDialog } from "./RateLimitResetDialog";
import { useUsageActions } from "@/services/usageExperience/useUsageActions";

interface Props {
  rateLimitMessage: string | null;
  failure?: ProviderFailure | null;
  sessionId?: string | null;
  onDismiss: () => void;
  onRefreshingChange?: (refreshing: boolean) => void;
  refreshState?: ReturnType<typeof useRateLimitRefresh>;
}

export function RateLimitGate({
  rateLimitMessage,
  failure = null,
  sessionId = null,
  onDismiss,
  onRefreshingChange,
  refreshState,
}: Props) {
  const state = useUsageActions();
  const localRefresh = useRateLimitRefresh(
    refreshState ? null : rateLimitMessage,
    state.retry,
    onRefreshingChange,
  );
  const refresh = refreshState ?? localRefresh;
  const { alternative, continueHere, isSwitching } = useProviderLimitDialog({
    failure,
    sessionId,
    onDismiss: () => {
      refresh.release();
      onDismiss();
    },
  });
  const failureWindow = /\b(weekly|daily|trial)\b/i
    .exec(rateLimitMessage ?? "")?.[1]
    ?.toLowerCase();
  function upgrade() {
    onDismiss();
    state.upgrade();
  }
  return (
    <RateLimitResetDialog
      isOpen={!!rateLimitMessage}
      onClose={onDismiss}
      experience={state.experience}
      offer={state.offer}
      onUpgrade={upgrade}
      checking={refresh.checking || state.isLoading}
      failureWindow={failureWindow}
      unavailable={state.isError || refresh.failed || !state.experience.blocked}
      onRetry={() => void refresh.refresh()}
      isBillingEnabled={state.isBillingEnabled}
      alternative={alternative}
      onContinue={continueHere}
      isSwitching={isSwitching}
    />
  );
}
