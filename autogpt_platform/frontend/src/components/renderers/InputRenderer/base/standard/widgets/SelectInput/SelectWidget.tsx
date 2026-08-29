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
import {
  MultiSelector,
  MultiSelectorContent,
  MultiSelectorInput,
  MultiSelectorItem,
  MultiSelectorList,
  MultiSelectorTrigger,
} from "@/components/__legacy__/ui/multiselect";

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
      "[SelectWidget] Dropped enum option(s) with empty-string value. Radix Select.Item disallows empty values.",
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
  const selectSize = size === "large" ? "medium" : "small";

  const renderInput = () => {
    if (type === InputType.MULTI_SELECT) {
      const enumOptionIndexesByLabel = new Map<string, string>();
      enumOptions.forEach((option, index) => {
        if (!enumOptionIndexesByLabel.has(option.label)) {
          enumOptionIndexesByLabel.set(option.label, String(index));
        }
      });

      const selectedValues: string[] = [];
      if (Array.isArray(selectedIndexes)) {
        for (const index of selectedIndexes) {
          const label = enumOptions[Number(index)]?.label;
          if (typeof label === "string") {
            selectedValues.push(label);
          }
        }
      }

      return (
        <MultiSelector
          values={selectedValues}
          onValuesChange={(newValues) => {
            const selectedOptionIndexes: string[] = [];
            for (const label of newValues) {
              const optionIndex = enumOptionIndexesByLabel.get(label);
              if (optionIndex !== undefined) {
                selectedOptionIndexes.push(optionIndex);
              }
            }
            onChange(
              enumOptionsValueForIndex(selectedOptionIndexes, enumOptions),
            );
          }}
          className="w-full"
        >
          <MultiSelectorTrigger>
            <MultiSelectorInput placeholder="Select options..." />
          </MultiSelectorTrigger>
          <MultiSelectorContent>
            <MultiSelectorList>
              {enumOptions.map((option) => (
                <MultiSelectorItem
                  key={`${String(option.value)}-${option.label}`}
                  value={option.label}
                >
                  {option.label}
                </MultiSelectorItem>
              ))}
            </MultiSelectorList>
          </MultiSelectorContent>
        </MultiSelector>
      );
    }
    const selectedValue =
      typeof selectedIndexes === "string" ? selectedIndexes : "";

    return (
      <Select
        label={resolvedLabel}
        placeholder={placeholder || schemaPlaceholder || "Select an option"}
        id={id}
        hideLabel={true}
        disabled={disabled || readonly}
        size={selectSize}
        value={selectedValue}
        onValueChange={(newValue) =>
          onChange(
            enumOptionsValueForIndex(newValue, enumOptions, options.emptyValue),
          )
        }
        options={enumOptions.map((option, index) => ({
          value: String(index),
          label: option.label,
        }))}
        wrapperClassName="!mb-0 "
        className={className}
      />
    );
  };

  return renderInput();
}
