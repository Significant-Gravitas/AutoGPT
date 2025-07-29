"use client";

import * as React from "react";
import { AgentSelectStep } from "./components/AgentSelectStep/AgentSelectStep";
import { AgentInfoStep } from "./components/AgentInfoStep/AgentInfoStep";
import { AgentReviewStep } from "./components/AgentReviewStep";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/atoms/Button/Button";
import { Props, usePublishAgentModal } from "./usePublishAgentModal";

export function PublishAgentModal({
  trigger,
  targetState,
  onStateChange,
}: Props) {
  const {
    // State
    currentState,
    updateState,
    initialData,
    // Handlers
    handleClose,
    handleAgentSelect,
    handleNextFromSelect,
    handleGoToDashboard,
    handleGoToBuilder,
    handleNextFromInfo,
    handleBack,
  } = usePublishAgentModal({ targetState, onStateChange });

  function renderContent() {
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
            onSubmit={handleNextFromInfo}
            initialData={initialData}
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
            onClose={handleClose}
            onDone={handleClose}
            onViewProgress={() => handleGoToDashboard()}
          />
        ) : (
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-1/2" />
          </div>
        );
    }
  }

  return (
    <>
      <Dialog
        styling={{
          maxWidth: "45rem",
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
        <Dialog.Trigger>
          {trigger || <Button size="small">Publish Agent</Button>}
        </Dialog.Trigger>
        <Dialog.Content>
          <div data-testid="publish-agent-modal">{renderContent()}</div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
