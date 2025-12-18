import { useMemo, useState } from "react";
import { RJSFSchema } from "@rjsf/utils";

const getDefaultValueForType = (type?: string): any => {
  if (!type) return "";

  switch (type) {
    case "string":
      return "";
    case "number":
    case "integer":
      return 0;
    case "boolean":
      return false;
    case "array":
      return [];
    case "object":
      return {};
    default:
      return "";
  }
};

export const useAnyOfField = (
  schema: RJSFSchema,
  formData: any,
  onChange: (value: any) => void,
) => {
  const typeOptions: any[] = useMemo(
    () =>
      schema.anyOf?.map((opt: any, i: number) => ({
        type: opt.type || "string",
        title: opt.title || `Option ${i + 1}`,
        index: i,
        format: opt.format,
        enum: opt.enum,
        secret: opt.secret,
        schema: opt,
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
    if (checked) {
      onChange(getDefaultValueForType(nonNull?.type));
    } else {
      onChange(null);
    }
  };

  const handleValueChange = (value: any) => onChange(value);

  const currentTypeOption = typeOptions.find((o) => o.type === selectedType);

  return {
    isNullableType,
    nonNull,
    selectedType,
    handleTypeChange,
    handleNullableToggle,
    handleValueChange,
    currentTypeOption,
    isEnabled,
    typeOptions,
  };
};
