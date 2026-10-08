import { useFieldAccessibility } from "../../../../field-accessibility";
import {
  enumOptionsIndexForValue,
  enumOptionsValueForIndex,
} from "@rjsf/utils";
import type { EnumOptionsType, RJSFSchema, WidgetProps } from "@rjsf/utils";
import {
  InputType,
  mapJsonSchemaTypeToInputType,
} from "@/app/(platform)/build/components/FlowEditor/nodes/helpers";
import { Select } from "@/components/atoms/Select/Select";
import { MultiSelect } from "@/components/molecules/MultiSelect/MultiSelect";

function isSchema(value: unknown): value is RJSFSchema {
  return typeof value === "object" && value !== null;
}

function getEnumNames(schema: RJSFSchema) {
  const candidates: unknown[] = [
    schema,
    ...(Array.isArray(schema.anyOf) ? schema.anyOf : []),
    ...(Array.isArray(schema.oneOf) ? schema.oneOf : []),
  ];
  for (const candidate of candidates) {
    if (isSchema(candidate) && Array.isArray(candidate.enumNames)) {
      return candidate.enumNames;
    }
  }
}

function getFieldSchema(props: WidgetProps) {
  const rootProperty = props.registry?.rootSchema?.properties?.[props.name];
  return isSchema(rootProperty) ? rootProperty : props.schema;
}

export function SelectWidget(props: WidgetProps) {
  const {
    options,
    value,
    onChange,
    disabled,
    readonly,
    className,
    id,
    formContext,
    label,
    placeholder,
  } = props;
  const rawEnumOptions: EnumOptionsType[] = options.enumOptions || [];
  const fieldSchema = getFieldSchema(props);
  const enumNames = getEnumNames(fieldSchema);
  const uiTitle = props.uiSchema?.["ui:title"];
  const resolvedLabel =
    typeof uiTitle === "string"
      ? uiTitle
      : typeof fieldSchema.title === "string"
        ? fieldSchema.title
        : label;
  const accessibility = useFieldAccessibility(
    id,
    resolvedLabel,
    formContext ?? props.registry?.formContext,
    props["aria-describedby"],
  );
  const schemaPlaceholder =
    typeof fieldSchema.placeholder === "string"
      ? fieldSchema.placeholder
      : undefined;
  const labelledEnumOptions = rawEnumOptions.map((option, index) =>
    Array.isArray(enumNames) && typeof enumNames[index] === "string"
      ? { ...option, label: enumNames[index] }
      : option,
  );
  const enumOptions = labelledEnumOptions.filter(
    (option) => option.value !== "",
  );
  const droppedEmptyOptionCount = rawEnumOptions.length - enumOptions.length;
  if (process.env.NODE_ENV === "development" && droppedEmptyOptionCount > 0) {
    console.warn(
      "[SelectWidget] Dropped enum option(s) with empty-string value. An empty value reads as no selection in Select.",
      {
        schema: props.schema,
        dropped: droppedEmptyOptionCount,
      },
    );
  }
  const type = mapJsonSchemaTypeToInputType(props.schema);
  const { size = "small" } = formContext || {};
  const selectedIndexes = enumOptionsIndexForValue(
    value,
    enumOptions,
    type === InputType.MULTI_SELECT,
  );

  // Determine select size based on context
  const selectSize = size === "large" ? "lg" : "md";

  const indexedOptions = enumOptions.map((option, index) => ({
    value: String(index),
    label: option.label,
  }));

  const renderInput = () => {
    if (type === InputType.MULTI_SELECT) {
      return (
        <MultiSelect
          {...accessibility}
          id={id}
          options={indexedOptions}
          value={Array.isArray(selectedIndexes) ? selectedIndexes : []}
          onValueChange={(indexes) =>
            onChange(enumOptionsValueForIndex(indexes, enumOptions))
          }
          disabled={disabled || readonly}
          placeholder="Select options..."
          className="w-full"
        />
      );
    }
    const selectedValue =
      typeof selectedIndexes === "string" ? selectedIndexes : "";

    return (
      <Select
        label={resolvedLabel}
        placeholder={placeholder || schemaPlaceholder || "Select an option"}
        {...accessibility}
        hideLabel={true}
        disabled={disabled || readonly}
        size={selectSize}
        value={selectedValue}
        onValueChange={(newValue) =>
          onChange(
            enumOptionsValueForIndex(newValue, enumOptions, options.emptyValue),
          )
        }
        options={indexedOptions}
        wrapperClassName="mb-0"
        className={className}
      />
    );
  };

  return renderInput();
}
