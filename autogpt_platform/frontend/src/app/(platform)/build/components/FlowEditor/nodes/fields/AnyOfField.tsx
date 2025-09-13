// autogpt_platform/frontend/src/app/(platform)/build/components/FlowEditor/CustomNode/fields/AnyOfField.tsx
import React, { useMemo, useState } from "react";
import { FieldProps } from "@rjsf/utils";

import { Text } from "@/components/atoms/Text/Text";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Select } from "@/components/atoms/Select/Select";
import { InputRenderer, InputType } from "../InputRenderer";

type TypeOption = {
  type: string;
  title: string;
  index: number;
  format?: string;
};

function resolveInputType(type?: string, format?: string): InputType {
  if (!type) return InputType.STRING;
  if (type === "string") {
    if (format === "date") return InputType.DATE;
    if (format === "time") return InputType.TIME;
    if (format === "date-time") return InputType.DATE_TIME;
    return InputType.STRING;
  }
  if (type === "number" || type === "integer") return InputType.NUMBER;
  if (type === "boolean") return InputType.BOOLEAN;
  if (type === "array") return InputType.ARRAY;
  if (type === "object") return InputType.OBJECT;
  return InputType.STRING;
}

export const AnyOfField = ({
  schema,
  formData,
  onChange,
  name,
}: FieldProps) => {
  const typeOptions: TypeOption[] = useMemo(
    () =>
      schema.anyOf?.map((opt: any, i: number) => ({
        type: opt.type || "string",
        title: opt.title || `Option ${i + 1}`,
        index: i,
        format: opt.format,
      })) || [],
    [schema.anyOf],
  );

  const isNullableType = useMemo(
    () =>
      typeOptions.length === 2 &&
      typeOptions.some((o) => o.type === "null") &&
      typeOptions.some((o) => o.type !== "null"),
    [typeOptions],
  );

  const nonNull = useMemo(
    () => (isNullableType ? typeOptions.find((o) => o.type !== "null") : null),
    [isNullableType, typeOptions],
  );

  const initialSelectedType = useMemo(() => {
    const def = schema.default;
    const first = typeOptions[0]?.type || "string";
    if (isNullableType) return nonNull?.type || "string";
    if (typeof def === "string" && typeOptions.some((o) => o.type === def))
      return def;
    return first;
  }, [schema.default, typeOptions, isNullableType, nonNull?.type]);

  const [selectedType, setSelectedType] = useState<string>(initialSelectedType);

  const isEnabled = formData !== null && formData !== undefined;

  const handleTypeChange = (t: string) => {
    setSelectedType(t);
    onChange(undefined); // clear current value when switching type
  };

  const handleNullableToggle = (checked: boolean) => {
    onChange(checked ? undefined : null);
  };

  const handleValueChange = (value: any) => onChange(value);

  const renderInput = (t: string, fmt?: string) => {
    const inputType = resolveInputType(t, fmt);
    return (
      <InputRenderer
        type={inputType}
        id={`${name}-input`}
        value={
          // Keep controlled inputs stable
          formData ?? (inputType === InputType.NUMBER ? "" : "")
        }
        placeholder={`Enter ${name}`}
        required={false}
        onChange={handleValueChange}
      />
    );
  };

  if (isNullableType) {
    return (
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1">
            <Text variant="body">
              {name.charAt(0).toUpperCase() + name.slice(1)}
            </Text>
            <Text variant="small" className="!text-green-500">
              ({nonNull?.type} | null)
            </Text>
          </div>
          <Switch checked={isEnabled} onCheckedChange={handleNullableToggle} />
        </div>
        {isEnabled && renderInput(nonNull?.type || "string", nonNull?.format)}
      </div>
    );
  }

  const currentFormat = typeOptions.find(
    (o) => o.type === selectedType,
  )?.format;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-1">
        <Text variant="body">
          {name.charAt(0).toUpperCase() + name.slice(1)}
        </Text>
        <Select
          label=""
          id={`${name}-type-select`}
          hideLabel={true}
          value={selectedType}
          onValueChange={handleTypeChange}
          options={typeOptions.map((o) => ({ value: o.type, label: o.type }))}
          size="small"
          wrapperClassName="!mb-0 "
          className="h-6 w-fit gap-1 pl-3 pr-2"
        />
      </div>
      {renderInput(selectedType, currentFormat)}
    </div>
  );
};
