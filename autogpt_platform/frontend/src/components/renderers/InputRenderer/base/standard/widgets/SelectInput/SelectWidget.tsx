import {
  enumOptionsIndexForValue,
  enumOptionsValueForIndex,
  WidgetProps,
} from "@rjsf/utils";
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
  } = props;
  const enumOptions = options.enumOptions || [];
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
      const selectedValues = Array.isArray(selectedIndexes)
        ? selectedIndexes
        : [];

      return (
        <MultiSelector
          values={selectedValues}
          onValuesChange={(newValues) =>
            onChange(enumOptionsValueForIndex(newValues, enumOptions))
          }
          className="w-full"
        >
          <MultiSelectorTrigger>
            <MultiSelectorInput placeholder="Select options..." />
          </MultiSelectorTrigger>
          <MultiSelectorContent>
            <MultiSelectorList>
              {enumOptions.map((option, index) => (
                <MultiSelectorItem
                  key={`${String(option.value)}-${option.label}`}
                  value={String(index)}
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
        label=""
        id={id}
        hideLabel={true}
        disabled={disabled || readonly}
        size={selectSize as any}
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
