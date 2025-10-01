"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { useState } from "react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useAgentRunModal } from "./useAgentRunModal";
import { ModalHeader } from "./components/ModalHeader/ModalHeader";
import { AgentCostSection } from "./components/AgentCostSection/AgentCostSection";
import { AgentSectionHeader } from "./components/AgentSectionHeader/AgentSectionHeader";
import { ModalRunSection } from "./components/ModalRunSection/ModalRunSection";
import { RunAgentModalContextProvider } from "./context";
import { AgentDetails } from "./components/AgentDetails/AgentDetails";
import { RunActions } from "./components/RunActions/RunActions";
import { ScheduleAgentModal } from "../ScheduleAgentModal/ScheduleAgentModal";
import { AlarmIcon } from "@phosphor-icons/react";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";

interface Props {
  triggerSlot: React.ReactNode;
  agent: LibraryAgent;
  agentId: string;
  agentVersion?: number;
  onRunCreated?: (execution: GraphExecutionMeta) => void;
  onScheduleCreated?: (schedule: GraphExecutionJobInfo) => void;
}

export function RunAgentModal({
  triggerSlot,
  agent,
  onRunCreated,
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
  });

  const [isScheduleModalOpen, setIsScheduleModalOpen] = useState(false);

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
                      <ModalRunSection />
                    </>
                  </RunAgentModalContextProvider>
                ) : null}
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
            <div className="flex items-center justify-end gap-3">
              <Button
                variant="secondary"
                onClick={handleOpenScheduleModal}
                disabled={
                  isExecuting || isSettingUpTrigger || !allRequiredInputsAreSet
                }
              >
                <AlarmIcon size={16} />
                Schedule Agent
              </Button>
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
