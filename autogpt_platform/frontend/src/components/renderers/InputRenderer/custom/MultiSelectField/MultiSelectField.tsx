import React from "react";
import { FieldProps, getUiOptions } from "@rjsf/utils";
import { BlockIOObjectSubSchema } from "@/lib/autogpt-server-api/types";
import {
  MultiSelector,
  MultiSelectorContent,
  MultiSelectorInput,
  MultiSelectorItem,
  MultiSelectorList,
  MultiSelectorTrigger,
} from "@/components/__legacy__/ui/multiselect";
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

  return (
    <div className={cn("flex flex-col", uiOptions.className)}>
      <MultiSelector
        className="nodrag"
        values={selection}
        onValuesChange={handleValuesChange}
      >
        <MultiSelectorTrigger className="rounded-3xl border border-zinc-200 bg-white px-2 shadow-none">
          <MultiSelectorInput
            placeholder={
              (schema as any).placeholder ?? `Select ${displayName}...`
            }
          />
        </MultiSelectorTrigger>
        <MultiSelectorContent className="nowheel">
          <MultiSelectorList>
            {options
              .map((key) => ({ ...optionSchema[key], key }))
              .map(({ key, title, description }) => (
                <MultiSelectorItem key={key} value={key} title={description}>
                  {title ?? key}
                </MultiSelectorItem>
              ))}
          </MultiSelectorList>
        </MultiSelectorContent>
      </MultiSelector>
    </div>
  );
};
