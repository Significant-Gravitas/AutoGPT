import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { BlockIORootSchema } from "@/lib/autogpt-server-api/types";
import { InputList } from "./RunnerInputList";

export interface BlockInput {
  id: string;
  inputSchema: BlockIORootSchema;
  hardcodedValues: {
    name: string;
    description: string;
    value: any;
    placeholder_values?: any[];
  };
}

interface RunSettingsUiProps {
  isOpen: boolean;
  onClose: () => void;
  blockInputs: BlockInput[];
  onInputChange: (nodeId: string, field: string, value: any) => void;
  onRun: () => void;
  onSchedule: () => Promise<void>;
  scheduledInput: boolean;
  isScheduling: boolean;
  isRunning: boolean;
}

export function RunnerInputUI({
  isOpen,
  onClose,
  blockInputs,
  isScheduling,
  onInputChange,
  onRun,
  onSchedule,
  scheduledInput,
  isRunning,
}: RunSettingsUiProps) {
  const handleRun = () => {
    onRun();
    onClose();
  };

  const handleSchedule = async () => {
    onClose();
    await onSchedule();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="flex flex-col px-10 py-8">
        <DialogHeader>
          <DialogTitle className="text-2xl">
            {scheduledInput ? "Schedule Settings" : "Run Settings"}
          </DialogTitle>
          <DialogDescription className="mt-2 text-sm">
            Configure settings for running your agent.
          </DialogDescription>
        </DialogHeader>
        <div className="flex-grow overflow-y-auto">
          <InputList blockInputs={blockInputs} onInputChange={onInputChange} />
        </div>
        <DialogFooter>
          <Button
            data-testid="run-dialog-run-button"
            onClick={scheduledInput ? handleSchedule : handleRun}
            className="text-lg"
            disabled={scheduledInput ? isScheduling : isRunning}
          >
            {scheduledInput ? "Schedule" : isRunning ? "Running..." : "Run"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default RunnerInputUI;
