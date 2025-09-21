import { WidgetProps } from "@rjsf/utils";
import { InputType, mapJsonSchemaTypeToInputType } from "../helpers";
import { Input } from "@/components/atoms/Input/Input";

export const TextInputWidget = (props: WidgetProps) => {
  const { schema } = props;
  const mapped = mapJsonSchemaTypeToInputType(schema);

  type InputConfig = {
    htmlType: string;
    placeholder: string;
    handleChange: (v: string) => any;
  };

  const inputConfig: Partial<Record<InputType, InputConfig>> = {
    [InputType.TEXT_AREA]: {
      htmlType: "textarea",
      placeholder: "Enter text...",
      handleChange: (v: string) => (v === "" ? undefined : v),
    },
    [InputType.PASSWORD]: {
      htmlType: "password",
      placeholder: "Enter secret text...",
      handleChange: (v: string) => (v === "" ? undefined : v),
    },
    [InputType.NUMBER]: {
      htmlType: "number",
      placeholder: "Enter number value...",
      handleChange: (v: string) => (v === "" ? undefined : Number(v)),
    },
    [InputType.INTEGER]: {
      htmlType: "account",
      placeholder: "Enter integer value...",
      handleChange: (v: string) => (v === "" ? undefined : Number(v)),
    },
  };

  const defaultConfig: InputConfig = {
    htmlType: "text",
    placeholder: "Enter string value...",
    handleChange: (v: string) => (v === "" ? undefined : v),
  };

  const config = (mapped && inputConfig[mapped]) || defaultConfig;

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    const v = e.target.value;
    return props.onChange(config.handleChange(v));
  };

  return (
    <Input
      id={props.id}
      hideLabel={true}
      type={config.htmlType as any}
      label={""}
      size="small"
      wrapperClassName="mb-0"
      value={props.value ?? ""}
      onChange={handleChange as any}
      placeholder={schema.placeholder || config.placeholder}
      required={props.required}
      disabled={props.disabled}
    />
  );
};
