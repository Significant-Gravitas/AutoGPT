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
import { useEffect, useRef, useState } from "react";
import { ScheduleAgentModal } from "../ScheduleAgentModal/ScheduleAgentModal";
import { AIAgentSafetyPopup } from "./components/AIAgentSafetyPopup/AIAgentSafetyPopup";
import { shouldShowSafetyPopup } from "./components/AIAgentSafetyPopup/helpers";
import { ModalHeader } from "./components/ModalHeader/ModalHeader";
import { ModalRunSection } from "./components/ModalRunSection/ModalRunSection";
import { RunActions } from "./components/RunActions/RunActions";
import { RunAgentModalContextProvider } from "./context";
import { useAgentRunModal } from "./useAgentRunModal";

interface Props {
  /** Opens the form; without one, the host opens it through `controlled`. */
  triggerSlot?: React.ReactNode;
  agent: LibraryAgent;
  initialInputValues?: Record<string, any>;
  initialTriggerConfigValues?: Record<string, any>;
  initialInputCredentials?: Record<string, any>;
  /** Credential fields left for the user to pick a key for, e.g. one whose
   *  key was deleted, each with the provider of the key it had: shown, and
   *  never filled for them. */
  fieldsToPick?: Record<string, string>;
  onRunCreated?: (execution: GraphExecutionMeta) => void;
  onTriggerSetup?: (preset: LibraryAgentPreset) => void;
  onScheduleCreated?: (schedule: GraphExecutionJobInfo) => void;
  /** Hosts that call agents something else (the expert page's workflows)
   *  override the dialog title. */
  title?: string;
  dialogVariant?: "default" | "compact";
  controlled?: { isOpen: boolean; set: (open: boolean) => void };
  /** Shown above the task's inputs. */
  notice?: React.ReactNode;
  /** The form repeats a run: an optional credential it starts without is one
   *  the run went without, so nothing fills it. */
  keepsSkippedCredentials?: boolean;
}

const NO_FIELDS_TO_PICK: Record<string, string> = {};

export function RunAgentModal({
  triggerSlot,
  agent,
  initialInputValues,
  initialTriggerConfigValues,
  initialInputCredentials,
  fieldsToPick = NO_FIELDS_TO_PICK,
  onRunCreated,
  onTriggerSetup,
  onScheduleCreated,
  title = "Run Workflow",
  dialogVariant = "default",
  controlled,
  notice,
  keepsSkippedCredentials,
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

    // Form: trigger config (the trigger node's config on a triggered agent)
    triggerConfigValues,
    setTriggerConfigValues,
    triggerConfigFields,

    // Form: credentials
    inputCredentials,
    handleCredentialChange,
    handleCredentialAutoSelect,
    clearedFields,
    scheduleCredentials,

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
    nodeCredentials,
    isResolvingCredentials,
    nodeCredentialsUnknown,
    isRetryingNodeCredentials,
    retryNodeCredentials,
    hasInputFields,

    // Async states
    isExecuting,
    isSettingUpTrigger,

    // Actions
    handleRun,
    handleSimulate,
  } = useAgentRunModal(agent, {
    onRun: onRunCreated,
    onSetupTrigger: onTriggerSetup,
    initialInputValues,
    initialTriggerConfigValues,
    initialInputCredentials,
    controlled,
    keepsSkippedCredentials,
  });

  const [isScheduleModalOpen, setIsScheduleModalOpen] = useState(false);
  const [hasOverflow, setHasOverflow] = useState(false);
  const [isSafetyPopupOpen, setIsSafetyPopupOpen] = useState(false);
  const [pendingRunAction, setPendingRunAction] = useState<(() => void) | null>(
    null,
  );
  const contentRef = useRef<HTMLDivElement>(null);

  const hasAnySetupFields =
    hasInputFields || Object.keys(agentCredentialsInputFields || {}).length > 0;

  const isTriggerRunType = defaultRunType.includes("trigger");

  useEffect(() => {
    if (!isOpen) return;

    function checkOverflow() {
      if (!contentRef.current) return;
      const scrollableParent = contentRef.current
        .closest("[data-dialog-content]")
        ?.querySelector('[class*="overflow-y-auto"]');
      if (scrollableParent) {
        setHasOverflow(
          scrollableParent.scrollHeight > scrollableParent.clientHeight,
        );
      }
    }

    const timeoutId = setTimeout(checkOverflow, 100);
    const resizeObserver = new ResizeObserver(checkOverflow);
    if (contentRef.current) {
      const scrollableParent = contentRef.current
        .closest("[data-dialog-content]")
        ?.querySelector('[class*="overflow-y-auto"]');
      if (scrollableParent) {
        resizeObserver.observe(scrollableParent);
      }
    }

    return () => {
      clearTimeout(timeoutId);
      resizeObserver.disconnect();
    };
  }, [
    isOpen,
    hasAnySetupFields,
    agentInputFields,
    agentCredentialsInputFields,
  ]);

  function handleInputChange(key: string, value: string) {
    setInputValues((prev) => ({
      ...prev,
      [key]: value,
    }));
  }

  function handleTriggerConfigChange(key: string, value: string) {
    setTriggerConfigValues((prev) => ({
      ...prev,
      [key]: value,
    }));
  }

  function handleSetOpen(open: boolean) {
    // The safety popup opens over the form and takes focus, which the form's
    // dialog reads as a click outside it. A host that unmounts the form once
    // it closes would take the popup, and the run it holds, with it.
    if (!open && isSafetyPopupOpen) return;
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

  function handleRunWithSafetyCheck() {
    if (shouldShowSafetyPopup(agent)) {
      setPendingRunAction(() => handleRun);
      setIsSafetyPopupOpen(true);
    } else {
      handleRun();
    }
  }

  function handleSafetyPopupAcknowledge() {
    setIsSafetyPopupOpen(false);
    if (pendingRunAction) {
      pendingRunAction();
      setPendingRunAction(null);
    }
  }

  return (
    <>
      <Dialog
        title={title}
        variant={dialogVariant}
        controlled={{ isOpen, set: handleSetOpen }}
        styling={{ maxWidth: "600px", maxHeight: "90vh" }}
      >
        {triggerSlot ? <Dialog.Trigger>{triggerSlot}</Dialog.Trigger> : null}
        <Dialog.Content>
          <div ref={contentRef} className="flex min-h-full flex-col">
            <div className="flex-1">
              {/* Header */}
              <ModalHeader agent={agent} />

              {notice ? <div className="mt-4">{notice}</div> : null}

              {/* Content */}
              {hasAnySetupFields ? (
                <div className="mt-4 pb-10">
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
                      triggerConfigValues,
                      setTriggerConfigValue: handleTriggerConfigChange,
                      triggerConfigFields,
                      inputCredentials,
                      setInputCredentialsValue: handleCredentialChange,
                      autoSelectInputCredentialsValue:
                        handleCredentialAutoSelect,
                      clearedCredentialFields: clearedFields,
                      credentialFieldsToPick: fieldsToPick,
                      agentCredentialsInputFields,
                      nodeCredentials,
                      isResolvingCredentials,
                      nodeCredentialsUnknown,
                      isRetryingNodeCredentials,
                      retryNodeCredentials,
                    }}
                  >
                    <ModalRunSection />
                  </RunAgentModalContextProvider>
                </div>
              ) : null}
            </div>

            <Dialog.Footer
              // oxlint-disable-next-line design-system/no-restyle -- a sticky footer over scrolling content needs its own fill and gap
              className={`sticky bottom-0 z-raised bg-surface pt-4 ${
                hasOverflow
                  ? // oxlint-disable-next-line design-system/no-restyle -- the overflow divider and its soft top shadow have no Dialog.Footer variant
                    "border-t border-separator shadow-scroll-edge"
                  : ""
              }`}
            >
              <div className="flex items-center justify-end gap-3">
                <RunActions
                  defaultRunType={defaultRunType}
                  onRun={handleRunWithSafetyCheck}
                  onSimulate={handleSimulate}
                  isExecuting={isExecuting}
                  isSettingUpTrigger={isSettingUpTrigger}
                  isRunReady={allRequiredInputsAreSet}
                  scheduleButton={
                    isTriggerRunType ? undefined : !allRequiredInputsAreSet ? (
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
                              Please set up all required inputs and credentials
                              before scheduling
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : (
                      <Button
                        variant="secondary"
                        onClick={handleOpenScheduleModal}
                        disabled={isExecuting || isSettingUpTrigger}
                      >
                        Schedule Task
                      </Button>
                    )
                  }
                />
              </div>
              <ScheduleAgentModal
                isOpen={isScheduleModalOpen}
                onClose={handleCloseScheduleModal}
                agent={agent}
                inputValues={inputValues}
                inputCredentials={scheduleCredentials}
                formCredentials={inputCredentials}
                onScheduleCreated={handleScheduleCreated}
              />
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>

      <AIAgentSafetyPopup
        agent={agent}
        isOpen={isSafetyPopupOpen}
        onAcknowledge={handleSafetyPopupAcknowledge}
      />
    </>
  );
}
