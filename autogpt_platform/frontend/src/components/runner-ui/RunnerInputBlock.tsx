import React from "react";

import {
  BlockIOStringSubSchema,
  BlockIOSubSchema,
} from "@/lib/autogpt-server-api";
import { TypeBasedInput } from "@/components/type-based-input";
import SchemaTooltip from "../SchemaTooltip";

interface InputBlockProps {
  name: string;
  schema: BlockIOSubSchema;
  description?: string;
  defaultValue?: any;
  placeholderValues?: any[];
  onInputChange: (value: any) => void;
}

export function InputBlock({
  name,
  schema,
  description,
  defaultValue,
  placeholderValues,
  onInputChange,
}: InputBlockProps) {
  if (placeholderValues && placeholderValues.length > 0) {
    schema = { ...schema, enum: placeholderValues } as BlockIOStringSubSchema;
  }

  return (
    <div className="space-y-2">
      <label className="flex items-center gap-1 text-sm font-medium">
        {name || "Unnamed Input"}
        <SchemaTooltip description={description} />
      </label>
      <TypeBasedInput
        data-testid={`run-dialog-input-${name}`}
        schema={schema}
        defaultValue={defaultValue}
        placeholder={description}
        onChange={onInputChange}
      />
    </div>
  );
}
