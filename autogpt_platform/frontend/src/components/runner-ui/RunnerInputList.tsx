import React from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { InputBlock } from "./RunnerInputBlock";
import { InputNodeInfo } from "./RunnerInputUI";

interface InputListProps {
  inputNodes: InputNodeInfo[];
  onInputChange: (nodeId: string, field: string, value: any) => void;
}

export function InputList({ inputNodes, onInputChange }: InputListProps) {
  return (
    <ScrollArea className="max-h-[60vh] overflow-auto">
      <div className="space-y-4">
        {inputNodes && inputNodes.length > 0 ? (
          inputNodes.map((inputNode) => (
            <InputBlock
              key={inputNode.id}
              id={inputNode.id}
              schema={inputNode.inputSchema}
              name={inputNode.inputConfig.name}
              description={inputNode.inputConfig.description}
              value={inputNode.inputConfig.defaultValue ?? ""}
              placeholder_values={inputNode.inputConfig.placeholderValues}
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
