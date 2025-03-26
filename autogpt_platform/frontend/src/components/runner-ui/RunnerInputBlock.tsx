import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import {
  BlockIOStringSubSchema,
  BlockIOSubSchema,
} from "@/lib/autogpt-server-api";
import { TypeBasedInput } from "@/components/TypeBasedInput";

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
    schema = schema as BlockIOStringSubSchema;
    schema.enum = placeholder_values;
  }

  return (
    <div className="space-y-1">
      <h3 className="text-base font-semibold">{name || "Unnamed Input"}</h3>
      {description && <p className="text-sm text-gray-600">{description}</p>}

      <TypeBasedInput
        schema={schema}
        value={value}
        onChange={(value) => onInputChange(id, "value", value)}
      />
    </div>
  );
}
