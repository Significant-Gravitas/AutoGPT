"use client";

import * as React from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { AgentSelectStep } from "./components/AgentSelectStep/AgentSelectStep";
import { AgentInfoStep } from "./components/AgentInfoStep/AgentInfoStep";
import { AgentReviewStep } from "./components/AgentReviewStep";
import { StepStrip } from "./components/StepStrip";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { Button } from "@/components/atoms/Button/Button";
import { Props, usePublishAgentModal } from "./usePublishAgentModal";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import {
  PublishAuthPrompt,
  PublishAuthPromptSkeleton,
} from "./components/PublishAuthPrompt";

export function PublishAgentModal({
  trigger,
  targetState,
  onStateChange,
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
    // Handlers
    handleClose,
    handleAgentSelect,
    handleNextFromSelect,
    handleGoToDashboard,
    handleGoToBuilder,
    handleSuccessFromInfo,
    handleBack,
  } = usePublishAgentModal({
    targetState,
    onStateChange,
    preSelectedAgentId,
    preSelectedAgentVersion,
  });

  const { user, isUserLoading } = useSupabase();
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
          />
        );
      case "info":
        return (
          <AgentInfoStep
            onBack={handleBack}
            onSuccess={handleSuccessFromInfo}
            selectedAgentId={selectedAgentId}
            selectedAgentVersion={selectedAgentVersion}
            initialData={initialData}
            isMarketplaceUpdate={!!currentState.submissionData}
          />
        );
      case "review":
        return currentState.submissionData &&
          currentState.submissionData.name ? (
          <AgentReviewStep
            agentName={currentState.submissionData.name}
            subheader={currentState.submissionData.sub_heading}
            description={currentState.submissionData.description || ""}
            thumbnailSrc={currentState.submissionData.image_urls?.[0]}
            status={currentState.submissionData.status}
            reviewComments={currentState.submissionData.review_comments}
            onClose={handleClose}
            onDone={handleClose}
            onViewProgress={() => handleGoToDashboard()}
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

  return (
    <>
      <Dialog
        styling={{
          maxWidth: "48rem",
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
            {trigger || <Button size="small">Publish Agent</Button>}
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
                  : { duration: 0.32, ease: [0.16, 1, 0.3, 1] },
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
                        duration: shouldReduceMotion ? 0 : 0.24,
                        ease: [0.16, 1, 0.3, 1],
                      },
                    },
                    exit: (dir: number) => ({
                      opacity: 0,
                      x: shouldReduceMotion ? 0 : dir * -40,
                      transition: {
                        duration: shouldReduceMotion ? 0 : 0.16,
                        ease: [0.4, 0, 1, 1],
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
