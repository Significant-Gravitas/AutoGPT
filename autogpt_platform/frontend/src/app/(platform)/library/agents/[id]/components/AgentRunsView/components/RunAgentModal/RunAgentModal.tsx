"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { useState } from "react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
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
    // UI state
    isOpen,
    setIsOpen,
    showScheduleView,

    // Run mode
    defaultRunType,

    // Form: regular inputs
    inputValues,
    setInputValues,

    // Form: credentials
    inputCredentials,
    setInputCredentials,

    // Preset/trigger labels
    presetName,
    presetDescription,
    setPresetName,
    setPresetDescription,

    // Scheduling
    scheduleName,
    cronExpression,

    // Validation/readiness
    allRequiredInputsAreSet,

    // Schemas
    agentInputFields,
    agentCredentialsInputFields,

    // Async states
    isExecuting,
    isCreatingSchedule,
    isSettingUpTrigger,

    // Actions
    handleRun,
    handleSchedule,
    handleShowSchedule,
    handleGoBack,
    handleSetScheduleName,
    handleSetCronExpression,
  } = useAgentRunModal(agent);

  const [isScheduleFormValid, setIsScheduleFormValid] = useState(true);

  const hasAnySetupFields =
    Object.keys(agentInputFields || {}).length > 0 ||
    Object.keys(agentCredentialsInputFields || {}).length > 0;

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
          <div className="flex h-full flex-col pb-4">
            {/* Header */}
            <div className="flex-shrink-0">
              <ModalHeader agent={agent} />
              <AgentCostSection flowId={agent.graph_id} />
            </div>

            {/* Scrollable content */}
            <div className="flex-1 pr-1" style={{ scrollbarGutter: "stable" }}>
              {/* Setup Section */}
              <div className="mt-10">
                {hasAnySetupFields ? (
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
              <div className="mt-4">
                <AgentSectionHeader title="Schedule Setup" />
                {showScheduleView ? (
                  <>
                    <div className="my-4 flex justify-start">
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
                      recommendedScheduleCron={agent.recommended_schedule_cron}
                      onScheduleNameChange={handleSetScheduleName}
                      onCronExpressionChange={handleSetCronExpression}
                      onValidityChange={setIsScheduleFormValid}
                    />
                  </>
                ) : (
                  <div className="mt-2 flex flex-col items-start gap-2">
                    <Text variant="body" className="mb-3 !text-zinc-500">
                      No schedule configured. Create a schedule to run this
                      agent automatically at a specific time.{" "}
                      {agent.recommended_schedule_cron && (
                        <span className="text-blue-600">
                          This agent has a recommended schedule.
                        </span>
                      )}
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
          </div>
          <Dialog.Footer
            className="fixed bottom-1 left-0 z-10 w-full bg-white p-4"
            style={{ boxShadow: "0px -8px 10px white" }}
          >
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
        </Dialog.Content>
      </Dialog>
    </>
  );
}
