"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { AttentionCard } from "@/components/organisms/AttentionCard/AttentionCard";
import { cn } from "@/lib/utils";
import { TrialCanceledDialog } from "./TrialCanceledDialog/TrialCanceledDialog";
import { TrialOffer } from "./TrialOffer";
import { TrialStatus } from "./TrialStatus";
import { useTrialCard } from "./useTrialCard";

interface Props {
  returnTo?: "onboarding" | "billing";
}

export function TrialCard({ returnTo = "billing" }: Props) {
  const controller = useTrialCard(returnTo);
  return <TrialCardContent returnTo={returnTo} controller={controller} />;
}

export function TrialCardContent({
  returnTo = "billing",
  controller,
}: Props & { controller: ReturnType<typeof useTrialCard> }) {
  const {
    userID,
    trial,
    isLoading,
    queryError,
    error,
    retry,
    isStarting,
    isBlocked,
    isCanceling,
    isResuming,
    startTrial,
    cancelTrial,
    resumeTrial,
    showCanceledDialog,
    dismissCanceledDialog,
    subscribeNow,
  } = controller;
  const isBilling = returnTo === "billing";
  if (isLoading)
    return (
      <Skeleton
        className={cn(
          "w-full",
          // TODO(tokens): no radius token for 18px.
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
          isBlocked={isBlocked}
          onStart={startTrial}
        />
      ) : (
        <TrialStatus
          key={userID}
          trial={trial}
          isCanceling={isCanceling}
          onCancel={cancelTrial}
          isResuming={isResuming}
          onResume={resumeTrial}
        />
      )}
      {error ? (
        <p
          role="alert"
          data-slot="trial-card-content-error"
          className="mt-3 text-sm text-destructive"
        >
          {error}
        </p>
      ) : null}
      <TrialCanceledDialog
        trial={trial}
        isOpen={showCanceledDialog}
        isResuming={isResuming}
        onResume={resumeTrial}
        onSubscribe={subscribeNow}
        onClose={dismissCanceledDialog}
      />
    </>
  );
  if (isBilling)
    return (
      <section
        aria-label="AutoGPT trial"
        data-slot="trial-card-content"
        className="flex w-full flex-col gap-2 py-4"
      >
        <div data-slot="trial-card-content-heading" className="sr-only">
          <Text variant="body-medium" as="span" className="text-textBlack">
            {trial.eligible ? "Free trial" : "Your plan"}
          </Text>
        </div>
        <div data-slot="trial-card-content-body" className="w-full">
          {body}
        </div>
      </section>
    );
  return (
    <AttentionCard.Root
      asChild
      variant="featured"
      density="roomy"
      className="w-full"
    >
      <section aria-label="AutoGPT trial" data-slot="trial-card-content">
        {body}
      </section>
    </AttentionCard.Root>
  );
}
