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

export interface InputNodeInfo {
  id: string;
  inputSchema: BlockIORootSchema;
  inputConfig: {
    name: string;
    description: string;
    defaultValue: any;
    placeholderValues?: any[];
  };
}

interface RunInputDialogProps {
  isOpen: boolean;
  doClose: () => void;
  inputs: InputNodeInfo[];
  onInputChange: (nodeId: string, field: string, value: any) => void;
  doRun: () => void;
  onSchedule: () => Promise<void>;
  scheduledInput: boolean;
  isScheduling: boolean;
  isRunning: boolean;
}

export function RunnerInputDialog({
  isOpen,
  doClose,
  inputs: inputNodes,
  isScheduling,
  onInputChange,
  doRun,
  onSchedule,
  scheduledInput,
  isRunning,
}: RunInputDialogProps) {
  const handleRun = () => {
    doRun();
    doClose();
  };

  const handleSchedule = async () => {
    doClose();
    await onSchedule();
  };

  return (
    <Dialog open={isOpen} onOpenChange={doClose}>
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
          <InputList inputNodes={inputNodes} onInputChange={onInputChange} />
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
