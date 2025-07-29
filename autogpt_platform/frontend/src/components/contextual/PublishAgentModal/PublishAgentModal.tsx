"use client";

import * as React from "react";
import { PublishAgentSelect } from "./components/PublishAgentSelect";
import { PublishAgentInfo } from "./components/PublishAgentInfo/PublishAgentInfo";
import { PublishAgentAwaitingReview } from "./components/PublishAgentAwaitingReview";
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
    currentState,
    updateState,
    initialData,
    myAgents,
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
          <PublishAgentSelect
            agents={
              myAgents?.agents
                .map((agent) => ({
                  name: agent.agent_name,
                  id: agent.agent_id,
                  version: agent.agent_version,
                  lastEdited: agent.last_edited,
                  imageSrc:
                    agent.agent_image || "https://picsum.photos/300/200",
                }))
                .sort(
                  (a, b) =>
                    new Date(b.lastEdited).getTime() -
                    new Date(a.lastEdited).getTime(),
                ) || []
            }
            onSelect={handleAgentSelect}
            onCancel={handleClose}
            onNext={handleNextFromSelect}
            onOpenBuilder={handleGoToBuilder}
          />
        );
      case "info":
        return (
          <PublishAgentInfo
            onBack={handleBack}
            onSubmit={handleNextFromInfo}
            initialData={initialData}
          />
        );
      case "review":
        return currentState.submissionData &&
          currentState.submissionData.name ? (
          <PublishAgentAwaitingReview
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
          maxWidth: "37rem",
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
        <Dialog.Content>{renderContent()}</Dialog.Content>
      </Dialog>
    </>
  );
}
