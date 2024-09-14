import React from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import { BlockIORootSchema } from "@/lib/autogpt-server-api/types";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

interface BlockOutput {
  id: string;
  outputSchema: BlockIORootSchema;
  hardcodedValues: {
    name: string;
    description: string;
  };
  result?: any;
}

interface OutputModalProps {
  isOpen: boolean;
  onClose: () => void;
  blockOutputs: BlockOutput[];
}

const formatOutput = (output: any): string => {
  if (typeof output === "object") {
    try {
      return JSON.stringify(output, null, 2);
    } catch (error) {
      return `Error formatting output: ${(error as Error).message}`;
    }
  }
  return String(output);
};

export function RunnerOutputUI({
  isOpen,
  onClose,
  blockOutputs,
}: OutputModalProps) {
  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent
        side="right"
        className="flex h-full w-full flex-col overflow-hidden sm:max-w-[500px]"
      >
        <SheetHeader className="px-2 py-2">
          <SheetTitle className="text-xl">Run Outputs</SheetTitle>
          <SheetDescription className="mt-1 text-sm">
            View the outputs from your agent run.
          </SheetDescription>
        </SheetHeader>
        <div className="flex-grow overflow-y-auto px-2 py-2">
          <ScrollArea className="h-full overflow-auto pr-4">
            <div className="space-y-4">
              {blockOutputs && blockOutputs.length > 0 ? (
                blockOutputs.map((block) => (
                  <div key={block.id} className="space-y-1">
                    <Label className="text-base font-semibold">
                      {block.hardcodedValues.name || "Unnamed Output"}
                    </Label>

                    {block.hardcodedValues.description && (
                      <Label className="block text-sm text-gray-600">
                        {block.hardcodedValues.description}
                      </Label>
                    )}

                    <div className="rounded-md bg-gray-100 p-2">
                      <Textarea
                        readOnly
                        value={formatOutput(block.result ?? "No output yet")}
                        className="resize-none whitespace-pre-wrap break-words border-none bg-transparent text-sm"
                      />
                    </div>
                  </div>
                ))
              ) : (
                <p>No output blocks available.</p>
              )}
            </div>
          </ScrollArea>
        </div>
      </SheetContent>
    </Sheet>
  );
}

export default RunnerOutputUI;
