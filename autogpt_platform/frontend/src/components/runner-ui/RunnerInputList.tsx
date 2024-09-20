import React from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { InputBlock } from "./RunnerInputBlock";
import { BlockInput } from "./RunnerInputUI";

interface InputListProps {
  blockInputs: BlockInput[];
  onInputChange: (nodeId: string, field: string, value: string) => void;
}

export function InputList({ blockInputs, onInputChange }: InputListProps) {
  return (
    <ScrollArea className="h-[20vh] overflow-auto pr-4 sm:h-[30vh] md:h-[40vh] lg:h-[50vh]">
      <div className="space-y-4">
        {blockInputs && blockInputs.length > 0 ? (
          blockInputs.map((block) => (
            <InputBlock
              key={block.id}
              id={block.id}
              name={block.hardcodedValues.name}
              description={block.hardcodedValues.description}
              value={block.hardcodedValues.value?.toString() || ""}
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
