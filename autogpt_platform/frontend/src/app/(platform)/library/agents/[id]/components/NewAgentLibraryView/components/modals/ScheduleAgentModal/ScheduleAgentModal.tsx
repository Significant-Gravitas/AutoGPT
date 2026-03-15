"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { useState } from "react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { ModalScheduleSection } from "./components/ModalScheduleSection/ModalScheduleSection";
import { Text } from "@/components/atoms/Text/Text";
import { useScheduleAgentModal } from "./useScheduleAgentModal";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  agent: LibraryAgent;
  inputValues: Record<string, any>;
  inputCredentials: Record<string, any>;
  onScheduleCreated?: (schedule: GraphExecutionJobInfo) => void;
}

export function ScheduleAgentModal({
  isOpen,
  onClose,
  agent,
  inputValues,
  inputCredentials,
  onScheduleCreated,
}: Props) {
  const [isScheduleFormValid, setIsScheduleFormValid] = useState(true);

  const {
    scheduleName,
    cronExpression,
    isCreatingSchedule,
    handleSchedule,
    handleSetScheduleName,
    handleSetCronExpression,
    resetForm,
  } = useScheduleAgentModal(agent, inputValues, inputCredentials, {
    onCreateSchedule: (schedule) => {
      onScheduleCreated?.(schedule);
    },
    onClose: onClose,
  });

  function handleClose() {
    resetForm();
    setIsScheduleFormValid(true);
    onClose();
  }

  async function handleScheduleClick() {
    if (!scheduleName.trim() || !isScheduleFormValid) return;

    try {
      await handleSchedule(scheduleName, cronExpression);
    } catch (error) {
      // Error handling is done in the hook
      console.error("Failed to create schedule:", error);
    }
  }

  const canSchedule = scheduleName.trim() && isScheduleFormValid;

  return (
    <Dialog
      controlled={{ isOpen, set: handleClose }}
      styling={{ maxWidth: "600px", maxHeight: "90vh" }}
    >
      <Dialog.Content>
        <div className="flex h-full flex-col">
          <Text variant="lead" as="h2" className="!font-medium !text-black">
            Schedule run
          </Text>

          {/* Content */}
          <div className="overflow-y-auto">
            <ModalScheduleSection
              scheduleName={scheduleName}
              cronExpression={cronExpression}
              recommendedScheduleCron={agent.recommended_schedule_cron}
              onScheduleNameChange={handleSetScheduleName}
              onCronExpressionChange={handleSetCronExpression}
              onValidityChange={setIsScheduleFormValid}
            />
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-3 pt-6">
            <Button
              variant="secondary"
              onClick={handleClose}
              disabled={isCreatingSchedule}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={handleScheduleClick}
              loading={isCreatingSchedule}
              disabled={!canSchedule}
            >
              Schedule
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
