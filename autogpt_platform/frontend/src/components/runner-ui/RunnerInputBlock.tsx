import React from "react";

import {
  BlockIOStringSubSchema,
  BlockIOSubSchema,
} from "@/lib/autogpt-server-api";
import { TypeBasedInput } from "@/components/type-based-input";
import SchemaTooltip from "../SchemaTooltip";

interface InputBlockProps {
  id: string;
  name: string;
  schema: BlockIOSubSchema;
  description?: string;
  value: string;
  placeholder_values?: any[];
  onInputChange: (id: string, field: string, value: string) => void;
}

export function InputBlock({
  id,
  name,
  schema,
  description,
  value,
  placeholder_values,
  onInputChange,
}: InputBlockProps) {
  if (placeholder_values && placeholder_values.length > 0) {
    schema = { ...schema, enum: placeholder_values } as BlockIOStringSubSchema;
  }

  return (
    <div className="space-y-2">
      <label className="flex items-center gap-1 text-sm font-medium">
        {name || "Unnamed Input"}
        <SchemaTooltip description={description} />
      </label>
      <TypeBasedInput
        id={`${id}-Value`}
        data-testid={`run-dialog-input-${name}`}
        schema={schema}
        value={value}
        placeholder={description}
        onChange={(value) => onInputChange(id, "value", value)}
      />
    </div>
  );
}
