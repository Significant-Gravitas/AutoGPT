import React from "react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface InputBlockProps {
  id: string;
  name: string;
  description?: string;
  value: string;
  placeholder_values?: any[];
  onInputChange: (id: string, field: string, value: string) => void;
}

export function InputBlock({
  id,
  name,
  description,
  value,
  placeholder_values,
  onInputChange,
}: InputBlockProps) {
  return (
    <div className="space-y-1">
      <h3 className="text-base font-semibold">{name || "Unnamed Input"}</h3>
      {description && <p className="text-sm text-gray-600">{description}</p>}
      <div>
        {placeholder_values && placeholder_values.length > 1 ? (
          <Select
            onValueChange={(value) => onInputChange(id, "value", value)}
            value={value}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a value" />
            </SelectTrigger>
            <SelectContent>
              {placeholder_values.map((placeholder, index) => (
                <SelectItem
                  key={index}
                  value={placeholder.toString()}
                  data-testid={`run-dialog-input-${name}-${placeholder.toString()}`}
                >
                  {placeholder.toString()}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : (
          <Input
            id={`${id}-Value`}
            data-testid={`run-dialog-input-${name}`}
            value={value}
            onChange={(e) => onInputChange(id, "value", e.target.value)}
            placeholder={placeholder_values?.[0]?.toString() || "Enter value"}
            className="w-full"
          />
        )}
      </div>
    </div>
  );
}
