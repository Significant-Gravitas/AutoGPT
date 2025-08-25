"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useAgentRunModal } from "./useAgentRunModal";
import { ModalHeader } from "./components/ModalHeader/ModalHeader";
import { AgentCostSection } from "./components/AgentCostSection/AgentCostSection";
import { AgentSectionHeader } from "./components/AgentSectionHeader/AgentSectionHeader";
import { DefaultRunView } from "./components/DefaultRunView/DefaultRunView";
import { ScheduleView } from "./components/ScheduleView/ScheduleView";
import { AgentDetails } from "./components/AgentDetails/AgentDetails";
import { RunActions } from "./components/RunActions/RunActions";
import { ScheduleActions } from "./components/ScheduleActions/ScheduleActions";

interface Props {
  triggerSlot: React.ReactNode;
  agent: LibraryAgent;
  agentId: string;
  agentVersion?: number;
}

export function RunAgentModal({ triggerSlot, agent }: Props) {
  const {
    isOpen,
    setIsOpen,
    showScheduleView,
    defaultRunType,
    inputValues,
    setInputValues,
    scheduleName,
    cronExpression,
    allRequiredInputsAreSet,
    // agentInputFields, // Available if needed for future use
    hasInputFields,
    isExecuting,
    isCreatingSchedule,
    isSettingUpTrigger,
    handleRun,
    handleSchedule,
    handleShowSchedule,
    handleGoBack,
    handleSetScheduleName,
    handleSetCronExpression,
  } = useAgentRunModal(agent);

  const [isScheduleFormValid, setIsScheduleFormValid] = useState(true);

  function handleInputChange(key: string, value: string) {
    setInputValues((prev) => ({
      ...prev,
      [key]: value,
    }));
  }

  function handleSetOpen(open: boolean) {
    setIsOpen(open);
    // Always reset to Run view when opening/closing
    if (open || !open) handleGoBack();
  }

  return (
    <Dialog
      controlled={{ isOpen, set: handleSetOpen }}
      styling={{ maxWidth: "600px", maxHeight: "90vh" }}
    >
      <Dialog.Trigger>{triggerSlot}</Dialog.Trigger>
      <Dialog.Content>
        <div className="flex h-full flex-col">
          {/* Header */}
          <div className="flex-shrink-0">
            <ModalHeader agent={agent} />
            <AgentCostSection flowId={agent.graph_id} />
          </div>

          {/* Scrollable content */}
          <div
            className="flex-1 overflow-y-auto overflow-x-hidden pr-1"
            style={{ scrollbarGutter: "stable" }}
          >
            {/* Setup Section */}
            <div className="mt-10">
              {showScheduleView ? (
                <>
                  <AgentSectionHeader title="Schedule Setup" />
                  <div>
                    <ScheduleView
                      agent={agent}
                      scheduleName={scheduleName}
                      cronExpression={cronExpression}
                      inputValues={inputValues}
                      onScheduleNameChange={handleSetScheduleName}
                      onCronExpressionChange={handleSetCronExpression}
                      onInputChange={handleInputChange}
                      onValidityChange={setIsScheduleFormValid}
                    />
                  </div>
                </>
              ) : hasInputFields ? (
                <>
                  <AgentSectionHeader
                    title={
                      defaultRunType === "automatic-trigger"
                        ? "Trigger Setup"
                        : "Agent Setup"
                    }
                  />
                  <div>
                    <DefaultRunView
                      agent={agent}
                      defaultRunType={defaultRunType}
                      inputValues={inputValues}
                      onInputChange={handleInputChange}
                    />
                  </div>
                </>
              ) : null}
            </div>

            {/* Agent Details Section */}
            <div className="mt-8">
              <AgentSectionHeader title="Agent Details" />
              <AgentDetails agent={agent} />
            </div>
          </div>

          {/* Fixed Actions - sticky inside dialog scroll */}
          <Dialog.Footer className="sticky bottom-0 z-10 bg-white">
            {!showScheduleView ? (
              <RunActions
                hasExternalTrigger={agent.has_external_trigger}
                defaultRunType={defaultRunType}
                onShowSchedule={handleShowSchedule}
                onRun={handleRun}
                isExecuting={isExecuting}
                isSettingUpTrigger={isSettingUpTrigger}
                allRequiredInputsAreSet={allRequiredInputsAreSet}
              />
            ) : (
              <ScheduleActions
                onGoBack={handleGoBack}
                onSchedule={handleSchedule}
                isCreatingSchedule={isCreatingSchedule}
                allRequiredInputsAreSet={
                  allRequiredInputsAreSet &&
                  !!scheduleName.trim() &&
                  isScheduleFormValid
                }
              />
            )}
          </Dialog.Footer>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
