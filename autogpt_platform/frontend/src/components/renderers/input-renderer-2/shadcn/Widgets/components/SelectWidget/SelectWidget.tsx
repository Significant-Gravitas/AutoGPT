import {
  FormContextType,
  RJSFSchema,
  StrictRJSFSchema,
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
import { ExtendedFormContextType } from "../../../types";
import {
  isAnyOfSchema,
  isAnyOfSelector,
} from "@/components/renderers/input-renderer-2/utils/schema-utils";
import { AnyOfSelector } from "./components/AnyOfSelector";

type CustomWidgetProps<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = ExtendedFormContextType,
> = WidgetProps<T, S, F> & {
  options: any;
};

export const SelectWidget = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = ExtendedFormContextType,
>(
  props: CustomWidgetProps<T, S, F>,
) => {
  const {
    options,
    value,
    onChange,
    schema,
    disabled,
    readonly,
    id,
    formContext,
  } = props;
  const enumOptions = options.enumOptions || [];
  const type = mapJsonSchemaTypeToInputType(props.schema);
  const { size = "small" } = formContext || {};

  const isAnyOfSelectorValue = isAnyOfSelector(id);
  if (isAnyOfSelectorValue) {
    return <AnyOfSelector {...props} />;
  }

  // Determine select size based on context
  const selectSize = size === "large" ? "medium" : "small";

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
              {enumOptions?.map((option: any) => (
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
        size={selectSize as any}
        value={value ?? ""}
        onValueChange={onChange}
        options={
          enumOptions?.map((option: any) => ({
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
