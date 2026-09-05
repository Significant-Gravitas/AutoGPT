"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { TrialOffer } from "./TrialOffer";
import { TrialStatus } from "./TrialStatus";
import { useTrialCard } from "./useTrialCard";

interface Props {
  returnTo?: "onboarding" | "billing";
}

export function TrialCard({ returnTo = "billing" }: Props) {
  const {
    trial,
    isLoading,
    queryError,
    error,
    retry,
    isStarting,
    isCanceling,
    startTrial,
    cancelTrial,
  } = useTrialCard(returnTo);
  if (isLoading) return <Skeleton className="h-36 w-full rounded-2xl" />;
  if (queryError)
    return <ErrorCard context="trial information" onRetry={retry} />;
  if (
    !trial?.offer ||
    trial.converted ||
    (!trial.eligible && trial.status === "checkout_pending")
  )
    return null;
  return (
    <section
      aria-label="AutoGPT trial"
      className="w-full rounded-2xl border border-border bg-background p-6 text-textBlack"
    >
      {trial.eligible ? (
        <TrialOffer
          trial={trial}
          isStarting={isStarting}
          onStart={startTrial}
        />
      ) : (
        <TrialStatus
          trial={trial}
          isCanceling={isCanceling}
          onCancel={cancelTrial}
        />
      )}
      {error ? (
        <p role="alert" className="mt-3 text-sm text-destructive">
          {error}
        </p>
      ) : null}
    </section>
  );
}
