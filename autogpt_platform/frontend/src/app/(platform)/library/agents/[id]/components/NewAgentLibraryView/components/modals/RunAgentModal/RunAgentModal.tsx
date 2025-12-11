"use client";

import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";
import { ScheduleAgentModal } from "../ScheduleAgentModal/ScheduleAgentModal";
import { ModalHeader } from "./components/ModalHeader/ModalHeader";
import { ModalRunSection } from "./components/ModalRunSection/ModalRunSection";
import { RunActions } from "./components/RunActions/RunActions";
import { RunAgentModalContextProvider } from "./context";
import { useAgentRunModal } from "./useAgentRunModal";

interface Props {
  triggerSlot: React.ReactNode;
  agent: LibraryAgent;
  initialInputValues?: Record<string, any>;
  initialInputCredentials?: Record<string, any>;
  onRunCreated?: (execution: GraphExecutionMeta) => void;
  onTriggerSetup?: (preset: LibraryAgentPreset) => void;
  onScheduleCreated?: (schedule: GraphExecutionJobInfo) => void;
}

export function RunAgentModal({
  triggerSlot,
  agent,
  initialInputValues,
  initialInputCredentials,
  onRunCreated,
  onTriggerSetup,
  onScheduleCreated,
}: Props) {
  const {
    // UI state
    isOpen,
    setIsOpen,

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

    // Validation/readiness
    allRequiredInputsAreSet,

    // Schemas
    agentInputFields,
    agentCredentialsInputFields,

    // Async states
    isExecuting,
    isSettingUpTrigger,

    // Actions
    handleRun,
  } = useAgentRunModal(agent, {
    onRun: onRunCreated,
    onSetupTrigger: onTriggerSetup,
    initialInputValues,
    initialInputCredentials,
  });

  const [isScheduleModalOpen, setIsScheduleModalOpen] = useState(false);

  const hasAnySetupFields =
    Object.keys(agentInputFields || {}).length > 0 ||
    Object.keys(agentCredentialsInputFields || {}).length > 0;

  const isTriggerRunType = defaultRunType.includes("trigger");

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
  }

  function handleOpenScheduleModal() {
    setIsScheduleModalOpen(true);
  }

  function handleCloseScheduleModal() {
    setIsScheduleModalOpen(false);
  }

  function handleScheduleCreated(schedule: GraphExecutionJobInfo) {
    handleCloseScheduleModal();
    setIsOpen(false); // Close the main RunAgentModal
    onScheduleCreated?.(schedule);
  }

  return (
    <>
      <Dialog
        controlled={{ isOpen, set: handleSetOpen }}
        styling={{ maxWidth: "600px", maxHeight: "90vh" }}
      >
        <Dialog.Trigger>{triggerSlot}</Dialog.Trigger>
        <Dialog.Content>
          {/* Header */}
          <ModalHeader agent={agent} />

          {/* Content */}
          {hasAnySetupFields ? (
            <div className="mt-10">
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
                <ModalRunSection />
              </RunAgentModalContextProvider>
            </div>
          ) : null}

          <Dialog.Footer className="mt-6 bg-white pt-4">
            <div className="flex items-center justify-end gap-3">
              {isTriggerRunType ? null : !allRequiredInputsAreSet ? (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span>
                        <Button
                          variant="secondary"
                          onClick={handleOpenScheduleModal}
                          disabled={
                            isExecuting ||
                            isSettingUpTrigger ||
                            !allRequiredInputsAreSet
                          }
                        >
                          Schedule Task
                        </Button>
                      </span>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>
                        Please set up all required inputs and credentials before
                        scheduling
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              ) : (
                <Button
                  variant="secondary"
                  onClick={handleOpenScheduleModal}
                  disabled={
                    isExecuting ||
                    isSettingUpTrigger ||
                    !allRequiredInputsAreSet
                  }
                >
                  Schedule Task
                </Button>
              )}
              <RunActions
                defaultRunType={defaultRunType}
                onRun={handleRun}
                isExecuting={isExecuting}
                isSettingUpTrigger={isSettingUpTrigger}
                isRunReady={allRequiredInputsAreSet}
              />
            </div>
            <ScheduleAgentModal
              isOpen={isScheduleModalOpen}
              onClose={handleCloseScheduleModal}
              agent={agent}
              inputValues={inputValues}
              inputCredentials={inputCredentials}
              onScheduleCreated={handleScheduleCreated}
            />
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
