"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { useState } from "react";
import { LibraryAgent } from "@/lib/autogpt-server-api/types";
import { useAgentRunModal } from "./useAgentRunModal";
import { ModalHeader } from "./components/ModalHeader/ModalHeader";
import { AgentCostSection } from "./components/AgentCostSection/AgentCostSection";
import { AgentSectionHeader } from "./components/AgentSectionHeader/AgentSectionHeader";
import { DefaultRunView } from "./components/DefaultRunView/DefaultRunView";
import { RunAgentModalContextProvider } from "./context";
import { ScheduleView } from "./components/ScheduleView/ScheduleView";
import { AgentDetails } from "./components/AgentDetails/AgentDetails";
import { RunActions } from "./components/RunActions/RunActions";
import { ScheduleActions } from "./components/ScheduleActions/ScheduleActions";
import { Text } from "@/components/atoms/Text/Text";
import { AlarmIcon, TrashIcon } from "@phosphor-icons/react";

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
    inputCredentials,
    setInputCredentials,
    presetName,
    presetDescription,
    setPresetName,
    setPresetDescription,
    scheduleName,
    cronExpression,
    allRequiredInputsAreSet,
    // agentInputFields, // Available if needed for future use
    agentInputFields,
    agentCredentialsInputFields,
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

  function handleCredentialsChange(key: string, value: any | undefined) {
    setInputCredentials((prev) => {
      const next = { ...prev } as Record<string, any>;
      if (value === undefined) {
        delete next[key];
        return next;
      }
      next[key] = value;
      return next;
    });
  }

  function handleSetOpen(open: boolean) {
    setIsOpen(open);
    // Always reset to Run view when opening/closing
    if (open || !open) handleGoBack();
  }

  function handleRemoveSchedule() {
    handleGoBack();
    handleSetScheduleName("");
    handleSetCronExpression("");
  }

  return (
    <>
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
                {hasInputFields ? (
                  <RunAgentModalContextProvider
                    value={{
                      agent,
                      defaultRunType,
                      presetName,
                      setPresetName,
                      presetDescription,
                      setPresetDescription,
                      inputValues,
                      setInputValue: handleInputChange,
                      agentInputFields,
                      inputCredentials,
                      setInputCredentialsValue: handleCredentialsChange,
                      agentCredentialsInputFields,
                    }}
                  >
                    <>
                      <AgentSectionHeader
                        title={
                          defaultRunType === "automatic-trigger"
                            ? "Trigger Setup"
                            : "Agent Setup"
                        }
                      />
                      <div>
                        <DefaultRunView />
                      </div>
                    </>
                  </RunAgentModalContextProvider>
                ) : null}
              </div>

              {/* Schedule Section - always visible */}
              <div className="mt-8">
                <AgentSectionHeader title="Schedule Setup" />
                {showScheduleView ? (
                  <>
                    <div className="mb-3 flex justify-start">
                      <Button
                        variant="secondary"
                        size="small"
                        onClick={handleRemoveSchedule}
                      >
                        <TrashIcon size={16} />
                        Remove schedule
                      </Button>
                    </div>
                    <ScheduleView
                      scheduleName={scheduleName}
                      cronExpression={cronExpression}
                      onScheduleNameChange={handleSetScheduleName}
                      onCronExpressionChange={handleSetCronExpression}
                      onValidityChange={setIsScheduleFormValid}
                    />
                  </>
                ) : (
                  <div className="flex flex-col items-start gap-2">
                    <Text variant="body" className="mb-3 !text-zinc-500">
                      No schedule configured. Create a schedule to run this
                      agent automatically at a specific time.{" "}
                    </Text>
                    <Button
                      variant="secondary"
                      size="small"
                      onClick={handleShowSchedule}
                    >
                      <AlarmIcon size={16} />
                      Create schedule
                    </Button>
                  </div>
                )}
              </div>

              {/* Agent Details Section */}
              <div className="mt-8">
                <AgentSectionHeader title="Agent Details" />
                <AgentDetails agent={agent} />
              </div>
            </div>

            {/* Fixed Actions - sticky inside dialog scroll */}
            <Dialog.Footer className="sticky bottom-0 z-10 bg-white">
              {showScheduleView ? (
                <ScheduleActions
                  onSchedule={handleSchedule}
                  isCreatingSchedule={isCreatingSchedule}
                  allRequiredInputsAreSet={
                    allRequiredInputsAreSet &&
                    !!scheduleName.trim() &&
                    isScheduleFormValid
                  }
                />
              ) : (
                <RunActions
                  defaultRunType={defaultRunType}
                  onRun={handleRun}
                  isExecuting={isExecuting}
                  isSettingUpTrigger={isSettingUpTrigger}
                  allRequiredInputsAreSet={allRequiredInputsAreSet}
                />
              )}
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
