import { useState } from "react";

interface Failure {
  userID: string;
  message: string;
}

interface ReportArgs {
  userID: string;
  error: unknown;
  fallback: string;
}

export function useTrialFailure(userID: string | undefined) {
  const [failure, setFailure] = useState<Failure | null>(null);

  function reportFailure({ userID, error, fallback }: ReportArgs) {
    setFailure({
      userID,
      message: error instanceof Error ? error.message : fallback,
    });
  }

  return {
    error: failure && failure.userID === userID ? failure.message : null,
    clearFailure: () => setFailure(null),
    reportFailure,
  };
}
