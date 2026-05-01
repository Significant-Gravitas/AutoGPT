interface ShouldShowSessionProcessingIndicatorArgs {
  sessionId: string;
  currentSessionId: string | null;
  isProcessing: boolean;
  hasCompletedIndicator: boolean;
  needsReload: boolean;
}

export function shouldShowSessionProcessingIndicator({
  sessionId,
  currentSessionId,
  isProcessing,
  hasCompletedIndicator,
  needsReload,
}: ShouldShowSessionProcessingIndicatorArgs) {
  return (
    isProcessing &&
    !needsReload &&
    !hasCompletedIndicator &&
    sessionId !== currentSessionId
  );
}
