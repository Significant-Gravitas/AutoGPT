"use client";

import * as React from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { AgentSelectStep } from "./components/AgentSelectStep/AgentSelectStep";
import { AgentInfoStep } from "./components/AgentInfoStep/AgentInfoStep";
import { AgentReviewStep } from "./components/AgentReviewStep";
import {
  ListingCheckFailed,
  ListingCheckPending,
} from "./components/ListingCheckState";
import { StepStrip } from "./components/StepStrip";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Button } from "@/components/atoms/Button/Button";
import { Props, usePublishAgentModal } from "./usePublishAgentModal";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { getApprovedMarketplaceUrl } from "@/lib/utils";
import {
  PublishAuthPrompt,
  PublishAuthPromptSkeleton,
} from "./components/PublishAuthPrompt";
import { duration, ease } from "@/components/styles/motion";

export function PublishAgentModal({
  trigger,
  targetState,
  onStateChange,
  onRequestEdit,
  preSelectedAgentId,
  preSelectedAgentVersion,
  showTrigger = true,
}: Props) {
  const {
    // State
    currentState,
    updateState,
    initialData,
    selectedAgentId,
    selectedAgentVersion,
    creatorUsername,
    isMarketplaceUpdate,
    isInfoFormReady,
    isCheckingListing,
    listingCheckFailed,
    // Handlers
    handleClose,
    handleAgentSelect,
    handleNextFromSelect,
    handleGoToDashboard,
    handleGoToBuilder,
    handleSuccessFromInfo,
    handleBack,
    handleRetryListingCheck,
  } = usePublishAgentModal({
    targetState,
    onStateChange,
    preSelectedAgentId,
    preSelectedAgentVersion,
  });

  const { user, isUserLoading } = useAuth();
  const shouldReduceMotion = useReducedMotion();

  const stepOrder = React.useMemo(
    () => ["select", "info", "review"] as const,
    [],
  );
  const currentStepIndex = stepOrder.indexOf(currentState.step);
  const previousStepIndexRef = React.useRef(currentStepIndex);
  const direction = currentStepIndex >= previousStepIndexRef.current ? 1 : -1;
  React.useEffect(() => {
    previousStepIndexRef.current = currentStepIndex;
  }, [currentStepIndex]);

  function renderContent() {
    if (isUserLoading) {
      return <PublishAuthPromptSkeleton />;
    }

    if (!user) {
      return <PublishAuthPrompt />;
    }

    switch (currentState.step) {
      case "select":
        return (
          <AgentSelectStep
            onSelect={handleAgentSelect}
            onCancel={handleClose}
            onNext={handleNextFromSelect}
            onOpenBuilder={handleGoToBuilder}
            isCheckingListing={isCheckingListing}
            listingCheckFailed={listingCheckFailed}
          />
        );
      case "info":
        if (!isInfoFormReady) {
          return listingCheckFailed ? (
            <ListingCheckFailed
              onRetry={handleRetryListingCheck}
              onCancel={handleClose}
            />
          ) : (
            <ListingCheckPending />
          );
        }
        return (
          <AgentInfoStep
            onBack={handleBack}
            onSuccess={handleSuccessFromInfo}
            selectedAgentId={selectedAgentId}
            selectedAgentVersion={selectedAgentVersion}
            initialData={initialData}
            isMarketplaceUpdate={isMarketplaceUpdate}
          />
        );
      case "review": {
        const submission = currentState.submissionData;
        const marketplaceUrl = getApprovedMarketplaceUrl({
          creatorUsername,
          slug: submission?.slug,
          isApproved: submission?.status === SubmissionStatus.APPROVED,
        });
        const onEdit =
          onRequestEdit &&
          submission &&
          submission.status !== SubmissionStatus.APPROVED
            ? () => onRequestEdit(submission)
            : undefined;
        return submission && submission.name ? (
          <AgentReviewStep
            agentName={submission.name}
            subheader={submission.sub_heading}
            description={submission.description || ""}
            thumbnailSrc={submission.image_urls?.[0]}
            status={submission.status}
            reviewComments={submission.review_comments}
            version={submission.listing_version}
            category={submission.categories?.[0]}
            submittedAt={submission.submitted_at}
            reviewedAt={submission.reviewed_at}
            runCount={submission.run_count}
            marketplaceUrl={marketplaceUrl}
            onClose={handleClose}
            onDone={handleClose}
            onViewProgress={() => handleGoToDashboard()}
            onEdit={onEdit}
          />
        ) : (
          <div className="flex min-h-[60vh] flex-col items-center justify-center gap-8 space-y-2">
            <Skeleton className="h-12 w-4/5" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        );
      }
    }
  }

  return (
    <>
      <Dialog
        title="Publish Workflow"
        styling={{
          maxWidth: "48rem",
          marginRight: "0.5rem",
        }}
        controlled={{
          isOpen: currentState.isOpen,
          set: (isOpen) => {
            if (!isOpen) {
              // When closing, always reset to clean state
              handleClose();
            } else {
              updateState({
                ...currentState,
                isOpen: isOpen,
              });
            }
          },
        }}
      >
        {showTrigger && (
          <Dialog.Trigger>
            {trigger || <Button size="small">Publish Workflow</Button>}
          </Dialog.Trigger>
        )}
        <Dialog.Content>
          <div data-testid="publish-agent-modal" className="min-h-0">
            {user && currentState.step && currentState.step !== "review" ? (
              <StepStrip currentStep={currentState.step} />
            ) : null}
            <motion.div
              layout
              transition={{
                layout: shouldReduceMotion
                  ? { duration: 0 }
                  : { duration: duration.slow, ease: ease.emphasized },
              }}
              className="overflow-hidden"
            >
              <AnimatePresence mode="wait" initial={false} custom={direction}>
                <motion.div
                  key={user ? currentState.step : "auth"}
                  custom={direction}
                  variants={{
                    enter: (dir: number) => ({
                      opacity: 0,
                      x: shouldReduceMotion ? 0 : dir * 40,
                    }),
                    center: {
                      opacity: 1,
                      x: 0,
                      transition: {
                        duration: shouldReduceMotion ? 0 : duration.base,
                        ease: ease.emphasized,
                      },
                    },
                    exit: (dir: number) => ({
                      opacity: 0,
                      x: shouldReduceMotion ? 0 : dir * -40,
                      transition: {
                        duration: shouldReduceMotion ? 0 : duration.fast,
                        ease: ease.in,
                      },
                    }),
                  }}
                  initial="enter"
                  animate="center"
                  exit="exit"
                >
                  {renderContent()}
                </motion.div>
              </AnimatePresence>
            </motion.div>
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
