import React from "react";
import { FieldProps, getUiOptions } from "@rjsf/utils";
import { BlockIOObjectSubSchema } from "@/lib/autogpt-server-api/types";
import { MultiSelect } from "@/components/molecules/MultiSelect/MultiSelect";
import { cn } from "@/lib/utils";
import { useMultiSelectField } from "./useMultiSelectField";

export const MultiSelectField = (props: FieldProps) => {
  const { schema, formData, onChange, fieldPathId } = props;
  const uiOptions = getUiOptions(props.uiSchema);

  const { optionSchema, options, selection, createChangeHandler } =
    useMultiSelectField({
      schema: schema as BlockIOObjectSubSchema,
      formData,
    });

  const handleValuesChange = createChangeHandler(onChange, fieldPathId);

  const displayName = schema.title || "options";
  const placeholder =
    typeof schema.placeholder === "string"
      ? schema.placeholder
      : `Select ${displayName}...`;

  return (
    <div className={cn("flex flex-col", uiOptions.className)}>
      <MultiSelect
        className="nodrag"
        aria-label={displayName}
        options={options.map((key) => ({
          value: key,
          label: optionSchema[key].title ?? key,
          description: optionSchema[key].description,
        }))}
        value={selection}
        onValueChange={handleValuesChange}
        placeholder={placeholder}
      />
    </div>
  );
};
