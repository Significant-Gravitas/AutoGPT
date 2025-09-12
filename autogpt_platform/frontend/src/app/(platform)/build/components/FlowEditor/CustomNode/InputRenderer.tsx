import { DateInput } from "@/components/atoms/DateInput/DateInput";
import { DateTimeInput } from "@/components/atoms/DateTimeInput/DateTimeInput";
import { Input } from "@/components/atoms/Input/Input";
import { Select, SelectOption } from "@/components/atoms/Select/Select";
import { Switch } from "@/components/atoms/Switch/Switch";
import { TimeInput } from "@/components/atoms/TimeInput/TimeInput";
import {
  MultiSelector,
  MultiSelectorContent,
  MultiSelectorInput,
  MultiSelectorItem,
  MultiSelectorList,
  MultiSelectorTrigger,
} from "@/components/ui/multiselect";
import { Input as SadcnInput } from "@/components/ui/input";

// These are all the types that we support in the input renderer
export enum InputType {
  STRING = "string",
  NUMBER = "number",
  BOOLEAN = "boolean",
  DATE = "date",
  TIME = "time",
  DATE_TIME = "datetime",
  FILE = "file",
  SELECT = "select",
  MULTI_SELECT = "multi-select",
  CREDENTIALS = "credentials",
  OBJECT = "object",
  ARRAY = "array",
}

export type InputRendererProps = {
  type: InputType;
  value: any;
  id: string;
  placeholder: string;
  required?: boolean;
  onChange: (value: any) => void;
  disabled?: boolean;
  readonly?: boolean;
  autofocus?: boolean;
  options?: SelectOption[];
  multiple?: boolean;
};

export const InputRenderer = (props: InputRendererProps) => {
  const {
    type,
    value,
    id,
    placeholder,
    required,
    onChange,
    disabled,
    readonly,
    autofocus,
    options,
    multiple,
  } = props;

  switch (type) {
    case InputType.STRING:
      return (
        <Input
          hideLabel={true}
          label={""}
          size="small"
          wrapperClassName="mb-0"
          id={id}
          value={value}
          onChange={onChange}
          placeholder={placeholder || ""}
          required={required}
        />
      );
    case InputType.NUMBER:
      return (
        <Input
          id={id}
          hideLabel={true}
          label={""}
          size="small"
          wrapperClassName="mb-0"
          value={value}
          onChange={onChange}
          placeholder={placeholder || ""}
          required={required}
        />
      );
    case InputType.BOOLEAN:
      return (
        <Switch
          checked={Boolean(value)}
          onCheckedChange={(checked) => onChange(checked)}
          disabled={disabled || readonly}
          autoFocus={autofocus}
        />
      );
    case InputType.DATE:
      return (
        <DateInput
          size="small"
          id={id}
          hideLabel={true}
          label={""}
          value={value}
          onChange={onChange}
          placeholder={placeholder || ""}
        />
      );
    case InputType.TIME:
      return (
        <TimeInput
          value={value}
          onChange={onChange}
          className="w-full"
          label={""}
          id={id}
          hideLabel={true}
          size="small"
          wrapperClassName="!mb-0 "
        />
      );
    case InputType.DATE_TIME:
      return (
        <DateTimeInput
          value={value}
          onChange={onChange}
          label={""}
          id={id}
          hideLabel={true}
          size="small"
          wrapperClassName="!mb-0 "
        />
      );
    case InputType.SELECT:
      return multiple ? (
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
              {options?.map((option) => (
                <MultiSelectorItem key={option.value} value={option.value}>
                  {option.label}
                </MultiSelectorItem>
              ))}
            </MultiSelectorList>
          </MultiSelectorContent>
        </MultiSelector>
      ) : (
        <Select
          label=""
          id={id}
          hideLabel={true}
          size="small"
          value={value}
          onValueChange={onChange}
          options={
            options?.map((option) => ({
              value: option.value,
              label: option.label,
            })) || []
          }
          wrapperClassName="!mb-0 "
        />
      );
    case InputType.FILE:
      // We need to work with the upload file function
      return (
        <SadcnInput
          id={id}
          type="file"
          multiple={multiple}
          disabled={disabled || readonly}
          onChange={onChange}
          className="rounded-full"
        />
      );
  }

  return null;
};
