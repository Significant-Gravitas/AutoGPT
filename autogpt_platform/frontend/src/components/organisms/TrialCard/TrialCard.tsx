"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";
import { TrialOffer } from "./TrialOffer";
import { TrialStatus } from "./TrialStatus";
import { useTrialCard } from "./useTrialCard";

interface Props {
  returnTo?: "onboarding" | "billing";
}

export function TrialCard({ returnTo = "billing" }: Props) {
  const {
    userID,
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
  const isBilling = returnTo === "billing";
  if (isLoading)
    return (
      <Skeleton
        className={cn(
          "w-full",
          isBilling ? "h-28 rounded-[18px]" : "h-36 rounded-2xl",
        )}
      />
    );
  if (queryError)
    return <ErrorCard context="trial information" onRetry={retry} />;
  if (
    !trial?.offer ||
    trial.converted ||
    (!trial.eligible && trial.status === "checkout_pending")
  )
    return null;
  const body = (
    <>
      {trial.eligible ? (
        <TrialOffer
          trial={trial}
          isStarting={isStarting}
          onStart={startTrial}
        />
      ) : (
        <TrialStatus
          key={userID}
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
    </>
  );
  if (isBilling)
    return (
      <section
        aria-label="AutoGPT trial"
        className="flex w-full flex-col gap-2"
      >
        <div className="flex items-center gap-2 px-4">
          <Text variant="body-medium" as="span" className="text-textBlack">
            {trial.eligible ? "Free trial" : "Your plan"}
          </Text>
        </div>
        <div className="rounded-[18px] border border-zinc-200 bg-white p-5 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
          {body}
        </div>
      </section>
    );
  return (
    <section
      aria-label="AutoGPT trial"
      className="w-full rounded-2xl bg-gradient-to-br from-zinc-300 via-zinc-400 to-zinc-500 p-px"
    >
      <div className="relative overflow-hidden rounded-[15px] bg-white p-5 md:p-6">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(120%_60%_at_0%_0%,rgba(168,85,247,0.10),transparent_60%),radial-gradient(120%_60%_at_100%_100%,rgba(99,102,241,0.06),transparent_60%)]"
        />
        <div className="relative">{body}</div>
      </div>
    </section>
  );
}
