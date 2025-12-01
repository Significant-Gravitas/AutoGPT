"use client";

import * as React from "react";
import { AgentSelectStep } from "./components/AgentSelectStep/AgentSelectStep";
import { AgentInfoStep } from "./components/AgentInfoStep/AgentInfoStep";
import { AgentReviewStep } from "./components/AgentReviewStep";
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
  } = usePublishAgentModal({ targetState, onStateChange });

  const { user, isUserLoading } = useSupabase();

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
