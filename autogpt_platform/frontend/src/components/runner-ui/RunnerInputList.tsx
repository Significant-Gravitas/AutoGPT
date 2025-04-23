import React from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { InputBlock } from "./RunnerInputBlock";
import { BlockInput } from "./RunnerInputUI";

interface InputListProps {
  blockInputs: BlockInput[];
  onInputChange: (nodeId: string, field: string, value: any) => void;
}

export function InputList({ blockInputs, onInputChange }: InputListProps) {
  return (
    <ScrollArea className="max-h-[60vh] overflow-auto">
      <div className="space-y-4">
        {blockInputs && blockInputs.length > 0 ? (
          blockInputs.map((block) => (
            <InputBlock
              key={block.id}
              id={block.id}
              schema={block.inputSchema}
              name={block.hardcodedValues.name}
              description={block.hardcodedValues.description}
              value={block.hardcodedValues.value || ""}
              placeholder_values={block.hardcodedValues.placeholder_values}
              onInputChange={onInputChange}
            />
          ))
        ) : (
          <p>No input blocks available.</p>
        )}
      </div>
    </ScrollArea>
  );
}
