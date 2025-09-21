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
import { ObjectEditor } from "../components/ObjectEditor/ObjectEditor";
import { ArrayEditor } from "../components/ArrayEditor/ArrayEditor";
import { ArrayFieldTemplateItemType, RJSFSchema } from "@rjsf/utils";

export enum InputType {
  SINGLE_LINE_TEXT = "single-line-text",
  TEXT_AREA = "text-area",
  PASSWORD = "password",
  FILE = "file",
  DATE = "date",
  TIME = "time",
  DATE_TIME = "datetime",
  NUMBER = "number",
  INTEGER = "integer",
  SWITCH = "switch",
  ARRAY_EDITOR = "array-editor",
  SELECT = "select",
  MULTI_SELECT = "multi-select",
  OBJECT_EDITOR = "object-editor",
  ENUM = "enum",
}

export type InputRendererProps = {
  type?: InputType;
  value: any;
  id: string;
  placeholder?: string;
  required?: boolean;
  onChange: (value: any) => void;
  disabled?: boolean;
  readonly?: boolean;
  autofocus?: boolean;
  options?: SelectOption[];
  multiple?: boolean;
  fieldKey?: string;
  nodeId?: string;

  // Array Editor Specific
  items?: ArrayFieldTemplateItemType<any, RJSFSchema, any>[];
  canAdd?: boolean | undefined;
  onAddClick?: () => void;
};

export const InputRenderer = (props: InputRendererProps) => {
  const {
    id,
    type,
    value,
    placeholder,
    required,
    onChange,
    disabled,
    readonly,
    autofocus,
    options,
    multiple,
    fieldKey,
    nodeId,
    items,
    canAdd,
    onAddClick,
  } = props;

  if (!type) return null;

  switch (type) {
    case InputType.SINGLE_LINE_TEXT:
      return (
        <Input
          id={id}
          hideLabel={true}
          label={""}
          size="small"
          wrapperClassName="mb-0"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder || ""}
          required={required}
        />
      );
    case InputType.TEXT_AREA:
      return (
        <Input
          hideLabel={true}
          label={""}
          size="small"
          type="textarea"
          wrapperClassName="mb-0"
          id={id}
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder || ""}
          required={required}
        />
      );
    case InputType.NUMBER:
      return (
        <Input
          id={id}
          type="number"
          hideLabel={true}
          label={""}
          size="small"
          wrapperClassName="mb-0"
          value={value ?? ""}
          onChange={(e) => {
            const v = e.target.value;
            onChange(v === "" ? undefined : Number(v));
          }}
          placeholder={placeholder || ""}
          required={required}
        />
      );
    case InputType.INTEGER:
      // Need to write better logic for integer input
      return (
        <Input
          id={id}
          type="amount"
          decimalCount={0}
          hideLabel={true}
          label={""}
          size="small"
          wrapperClassName="mb-0"
          value={value ?? ""}
          onChange={(e) => {
            const v = e.target.value;
            onChange(v === "" ? undefined : Number(v));
          }}
          placeholder={placeholder || ""}
          required={required}
        />
      );
    case InputType.SWITCH:
      return (
        <Switch
          id={id}
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
    case InputType.MULTI_SELECT:
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
              {options?.map((option) => (
                <MultiSelectorItem key={option.value} value={option.value}>
                  {option.label}
                </MultiSelectorItem>
              ))}
            </MultiSelectorList>
          </MultiSelectorContent>
        </MultiSelector>
      );
    case InputType.FILE:
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
    case InputType.PASSWORD:
      return (
        <Input
          hideLabel={true}
          label={""}
          id={id}
          type="password"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder || ""}
          required={required}
          wrapperClassName="!mb-0 nodrag"
        />
      );
    case InputType.SELECT:
      return (
        <Select
          label=""
          id={id}
          hideLabel={true}
          size="small"
          value={value ?? ""}
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
    case InputType.OBJECT_EDITOR:
      return (
        <ObjectEditor
          nodeId={nodeId ?? ""}
          fieldKey={fieldKey ?? ""}
          value={value}
          onChange={onChange}
        />
      );

    case InputType.ARRAY_EDITOR:
      console.log("input type array editor", items);
      return (
        <ArrayEditor
          nodeId={nodeId ?? ""}
          id={fieldKey ?? ""}
          items={items}
          canAdd={canAdd}
          onAddClick={onAddClick}
          disabled={disabled}
          readonly={readonly}
        />
      );
  }

  return null;
};
