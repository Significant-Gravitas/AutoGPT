"use client";

import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { AlarmIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { ScheduleAgentModal } from "../ScheduleAgentModal/ScheduleAgentModal";
import { AgentCostSection } from "./components/AgentCostSection/AgentCostSection";
import { AgentDetails } from "./components/AgentDetails/AgentDetails";
import { AgentSectionHeader } from "./components/AgentSectionHeader/AgentSectionHeader";
import { ModalHeader } from "./components/ModalHeader/ModalHeader";
import { ModalRunSection } from "./components/ModalRunSection/ModalRunSection";
import { RunActions } from "./components/RunActions/RunActions";
import { RunAgentModalContextProvider } from "./context";
import { useAgentRunModal } from "./useAgentRunModal";

interface Props {
  triggerSlot: React.ReactNode;
  agent: LibraryAgent;
  agentVersion?: number;
  initialInputValues?: Record<string, any>;
  initialInputCredentials?: Record<string, any>;
  initialPresetName?: string;
  initialPresetDescription?: string;
  onRunCreated?: (execution: GraphExecutionMeta) => void;
  onTriggerSetup?: (preset: LibraryAgentPreset) => void;
  onScheduleCreated?: (schedule: GraphExecutionJobInfo) => void;
  editMode?: {
    preset: LibraryAgentPreset;
    onSaved?: (updatedPreset: LibraryAgentPreset) => void;
  };
}

export function RunAgentModal({
  triggerSlot,
  agent,
  initialInputValues,
  initialInputCredentials,
  initialPresetName,
  initialPresetDescription,
  onRunCreated,
  onTriggerSetup,
  onScheduleCreated,
  editMode,
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

    // Edit mode
    hasChanges,
    isUpdatingPreset,
    handleSave,

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
    onScheduleCreated,
    initialInputValues,
    initialInputCredentials,
    initialPresetName,
    initialPresetDescription,
    editMode,
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

  const templateOrTrigger = agent.trigger_setup_info ? "Trigger" : "Template";

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
              {/* Template Info Section (Edit Mode Only) */}
              {editMode && (
                <div className="mt-10">
                  <AgentSectionHeader
                    title={`${templateOrTrigger} Information`}
                  />
                  <div className="mb-10 mt-4 space-y-4">
                    <div className="flex flex-col space-y-2">
                      <label className="text-sm font-medium">
                        {templateOrTrigger} Name
                      </label>
                      <Input
                        id="template_name"
                        label="Template Name"
                        size="small"
                        hideLabel
                        value={presetName}
                        placeholder="Enter template name"
                        onChange={(e) => setPresetName(e.target.value)}
                      />
                    </div>
                    <div className="flex flex-col space-y-2">
                      <label className="text-sm font-medium">
                        {templateOrTrigger} Description
                      </label>
                      <Input
                        id="template_description"
                        label="Template Description"
                        size="small"
                        hideLabel
                        value={presetDescription}
                        placeholder="Enter template description"
                        onChange={(e) => setPresetDescription(e.target.value)}
                      />
                    </div>
                  </div>
                </div>
              )}
              {/* Setup Section */}
              <div className={editMode ? "mt-8" : "mt-10"}>
                {hasAnySetupFields ? (
                  <RunAgentModalContextProvider
                    value={{
                      agent,
                      defaultRunType,
                      inputValues,
                      setInputValue: handleInputChange,
                      agentInputFields,
                      inputCredentials,
                      setInputCredentialsValue: handleCredentialsChange,
                      agentCredentialsInputFields,
                      presetEditMode: Boolean(
                        editMode || agent.trigger_setup_info,
                      ),
                      presetName,
                      setPresetName,
                      presetDescription,
                      setPresetDescription,
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
              {editMode ? (
                <>
                  <Button
                    variant="secondary"
                    onClick={() => setIsOpen(false)}
                    disabled={isUpdatingPreset}
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="primary"
                    onClick={handleSave}
                    disabled={
                      !hasChanges || isUpdatingPreset || !presetName.trim()
                    }
                  >
                    {isUpdatingPreset ? "Saving..." : "Save Changes"}
                  </Button>
                </>
              ) : (
                <>
                  {(defaultRunType == "manual" ||
                    defaultRunType == "schedule") && (
                    <Button
                      variant="secondary"
                      onClick={handleOpenScheduleModal}
                      disabled={
                        isExecuting ||
                        isSettingUpTrigger ||
                        !allRequiredInputsAreSet
                      }
                    >
                      <AlarmIcon size={16} />
                      Schedule Agent
                    </Button>
                  )}
                  <RunActions
                    defaultRunType={defaultRunType}
                    onRun={handleRun}
                    isExecuting={isExecuting}
                    isSettingUpTrigger={isSettingUpTrigger}
                    isRunReady={allRequiredInputsAreSet}
                  />
                </>
              )}
            </div>
            {(defaultRunType == "manual" || defaultRunType == "schedule") && (
              <ScheduleAgentModal
                isOpen={isScheduleModalOpen}
                onClose={handleCloseScheduleModal}
                agent={agent}
                inputValues={inputValues}
                inputCredentials={inputCredentials}
                onScheduleCreated={handleScheduleCreated}
              />
            )}
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
