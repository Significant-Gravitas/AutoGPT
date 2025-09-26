import { WidgetProps } from "@rjsf/utils";
import { InputType, mapJsonSchemaTypeToInputType } from "../helpers";
import { Select } from "@/components/atoms/Select/Select";
import {
  MultiSelector,
  MultiSelectorContent,
  MultiSelectorInput,
  MultiSelectorItem,
  MultiSelectorList,
  MultiSelectorTrigger,
} from "@/components/__legacy__/ui/multiselect";

export const SelectWidget = (props: WidgetProps) => {
  const { options, value, onChange, disabled, readonly, id } = props;
  const enumOptions = options.enumOptions || [];
  const type = mapJsonSchemaTypeToInputType(props.schema);

  const renderInput = () => {
    if (type === InputType.MULTI_SELECT) {
      return (
        <MultiSelector
          values={Array.isArray(value) ? value : []}
          onValuesChange={onChange}
          className="w-full"
        >
          <MultiSelectorTrigger>
            <MultiSelectorInput placeholder="Select options..." />
          </MultiSelectorTrigger>
          <MultiSelectorContent>
            <MultiSelectorList>
              {enumOptions?.map((option) => (
                <MultiSelectorItem key={option.value} value={option.value}>
                  {option.label}
                </MultiSelectorItem>
              ))}
            </MultiSelectorList>
          </MultiSelectorContent>
        </MultiSelector>
      );
    }
    return (
      <Select
        label=""
        id={id}
        hideLabel={true}
        disabled={disabled || readonly}
        size="small"
        value={value ?? ""}
        onValueChange={onChange}
        options={
          enumOptions?.map((option) => ({
            value: option.value,
            label: option.label,
          })) || []
        }
        wrapperClassName="!mb-0 "
      />
    );
  };

  return renderInput();
};
