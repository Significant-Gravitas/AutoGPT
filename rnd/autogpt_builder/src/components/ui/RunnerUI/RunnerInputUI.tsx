import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { BlockIORootSchema } from "@/lib/autogpt-server-api/types";

interface BlockInput {
  id: string;
  inputSchema: BlockIORootSchema;
  hardcodedValues: {
    name: string;
    description: string;
    value: any;
    placeholder_values?: any[];
    limit_to_placeholder_values?: boolean;
  };
}

interface RunSettingsUiProps {
  isOpen: boolean;
  onClose: () => void;
  blockInputs: BlockInput[];
  onInputChange: (nodeId: string, field: string, value: string) => void;
  onRun: () => void;
  isRunning: boolean;
}

export function RunnerInputUI({
  isOpen,
  onClose,
  blockInputs,
  onInputChange,
  onRun,
  isRunning,
}: RunSettingsUiProps) {
  const inputCount = blockInputs.length;

  const getDialogSize = (count: number) => {
    const baseWidth = 400;
    const widthPerInput = 50;
    const maxWidth = 600;
    return `sm:max-w-[${Math.min(baseWidth + count * widthPerInput, maxWidth)}px]`;
  };

  const getScrollAreaHeight = (count: number) => {
    const baseHeight = 20;
    const heightPerInput = 5;
    const maxHeight = 50;
    return `h-[${Math.min(baseHeight + count * heightPerInput, maxHeight)}vh]`;
  };

  const dialogSize = getDialogSize(inputCount);
  const scrollAreaHeight = getScrollAreaHeight(inputCount);

  // Auto close on run
  const handleRun = () => {
    onRun();
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent
        className={`flex max-h-[80vh] flex-col overflow-hidden ${dialogSize}`}
      >
        <DialogHeader className="px-4 py-4">
          <DialogTitle className="text-2xl">Run Settings</DialogTitle>
          <DialogDescription className="mt-2 text-sm">
            Configure settings for running your agent.
          </DialogDescription>
        </DialogHeader>
        <div className="flex-grow overflow-y-auto px-4 py-4">
          <ScrollArea className={`${scrollAreaHeight} overflow-auto pr-4`}>
            <div className="space-y-4">
              {blockInputs && blockInputs.length > 0 ? (
                blockInputs.map((block) => (
                  <div key={block.id} className="space-y-1">
                    <h3 className="text-base font-semibold">
                      {block.hardcodedValues.name || "Unnamed Input"}
                    </h3>

                    {block.hardcodedValues.description && (
                      <p className="text-sm text-gray-600">
                        {block.hardcodedValues.description}
                      </p>
                    )}

                    <div>
                      {block.hardcodedValues.placeholder_values &&
                      block.hardcodedValues.placeholder_values.length > 1 ? (
                        <Select
                          onValueChange={(value) =>
                            onInputChange(block.id, "value", value)
                          }
                          value={block.hardcodedValues.value?.toString() || ""}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select a value" />
                          </SelectTrigger>
                          <SelectContent>
                            {block.hardcodedValues.placeholder_values.map(
                              (placeholder, index) => (
                                <SelectItem
                                  key={index}
                                  value={placeholder.toString()}
                                >
                                  {placeholder.toString()}
                                </SelectItem>
                              ),
                            )}
                          </SelectContent>
                        </Select>
                      ) : (
                        <Input
                          id={`${block.id}-Value`}
                          value={block.hardcodedValues.value?.toString() || ""}
                          onChange={(e) =>
                            onInputChange(block.id, "value", e.target.value)
                          }
                          placeholder={
                            block.hardcodedValues.placeholder_values?.[0]?.toString() ||
                            "Enter value"
                          }
                          className="w-full"
                        />
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <p>No input blocks available.</p>
              )}
            </div>
          </ScrollArea>
        </div>
        <DialogFooter className="px-6 py-4">
          <Button
            onClick={handleRun}
            className="px-8 py-2 text-lg"
            disabled={isRunning}
          >
            {isRunning ? "Running..." : "Run"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default RunnerInputUI;
