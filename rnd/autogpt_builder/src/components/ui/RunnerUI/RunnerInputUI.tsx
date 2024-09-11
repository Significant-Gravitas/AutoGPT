import React from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
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
  onInputChange: (nodeId: string, field: string, value: any) => void;
  onRun: () => void;
  isRunning: boolean;
}

export function RunnerInputUI({ isOpen, onClose, blockInputs, onInputChange, onRun }: RunSettingsUiProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[500px] max-h-[80vh] flex flex-col overflow-hidden">
        <DialogHeader className="py-2 px-2">
          <DialogTitle className="text-xl">Run Settings</DialogTitle>
          <DialogDescription className="text-sm mt-1">
            Configure settings for running your agent.
          </DialogDescription>
        </DialogHeader>
        <div className="flex-grow overflow-y-auto px-2 py-2">
          <ScrollArea className="h-[50vh] pr-4 overflow-auto">
            <div className="space-y-4">
              {blockInputs && blockInputs.length > 0 ? (
                blockInputs.map((block) => (
                  <div key={block.id} className="space-y-1">
                    <h3 className="font-semibold text-base">{block.hardcodedValues.name || 'Unnamed Input'}</h3>
                    
                    {block.hardcodedValues.description && (
                      <p className="text-sm text-gray-600">{block.hardcodedValues.description}</p>
                    )}

                    <div>
                      {block.hardcodedValues.placeholder_values && block.hardcodedValues.placeholder_values.length > 1 ? (
                        <Select 
                          onValueChange={(value) => onInputChange(block.id, 'value', value)}
                          value={block.hardcodedValues.value?.toString() || ''}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select a value" />
                          </SelectTrigger>
                          <SelectContent>
                            {block.hardcodedValues.placeholder_values.map((placeholder, index) => (
                              <SelectItem key={index} value={placeholder.toString()}>
                                {placeholder.toString()}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      ) : (
                        <Input
                          id={`${block.id}-Value`}
                          value={block.hardcodedValues.value?.toString() || ''}
                          onChange={(e) => onInputChange(block.id, 'value', e.target.value)}
                          placeholder={block.hardcodedValues.placeholder_values?.[0]?.toString() || "Enter value"}
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
        <DialogFooter className="py-2 px-6">
          <Button onClick={onRun} className="px-6 py-2">
            {"Run"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default RunnerInputUI;