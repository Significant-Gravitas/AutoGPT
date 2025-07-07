import React from "react";

import { type BlockIORootSchema } from "@/lib/autogpt-server-api/types";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { isEmpty } from "@/lib/utils";
import { useToast } from "@/components/ui/use-toast";
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
  doRun: (inputs: Record<string, any>) => void;
  doCreateSchedule: () => void;
  scheduledInput: boolean;
  isScheduling: boolean;
  isRunning: boolean;
}

export function RunnerInputDialog({
  isOpen,
  doClose,
  inputs,
  isScheduling,
  doRun,
  doCreateSchedule,
  scheduledInput,
  isRunning,
}: RunInputDialogProps) {
  const { toast } = useToast();
  const [inputValues, setInputValues] = React.useState<Record<string, any>>({});

  const handleInputChange = (inputName: string, value: any) => {
    setInputValues((prev) => ({
      ...prev,
      [inputName]: value,
    }));
  };
  const requiredInputs = new Set(
    inputs
      .filter((input) => !input.inputConfig.defaultValue)
      .map((input) => input.inputConfig.name),
  );
  const missingInputs = requiredInputs.difference(
    new Set(Object.keys(inputValues).filter((k) => !isEmpty(inputValues[k]))),
  );
  const notifyMissingInputs = () => {
    toast({
      title: "⚠️ Not all required inputs are set",
      description: `Please set ${Array.from(missingInputs)
        .map((k) => `\`${k}\``)
        .join(", ")}`,
    });
  };

  const handleRun = () => {
    if (missingInputs.size > 0) {
      notifyMissingInputs();
      return;
    }
    doRun(inputValues);
    doClose();
  };

  const handleSchedule = async () => {
    if (missingInputs.size > 0) {
      notifyMissingInputs();
      return;
    }
    doClose();
    doCreateSchedule();
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
          <InputList inputNodes={inputs} onInputChange={handleInputChange} />
        </div>
        <DialogFooter>
          {scheduledInput ? (
            <Button
              data-testid="run-dialog-schedule-button"
              onClick={handleSchedule}
              className="text-lg"
              disabled={isScheduling}
            >
              Schedule
            </Button>
          ) : (
            <Button
              data-testid="run-dialog-run-button"
              onClick={handleRun}
              className="text-lg"
              disabled={isRunning}
            >
              {isRunning ? "Running..." : "Run"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
