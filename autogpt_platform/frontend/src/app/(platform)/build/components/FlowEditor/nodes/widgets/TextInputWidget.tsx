import { WidgetProps } from "@rjsf/utils";
import { InputType, mapJsonSchemaTypeToInputType } from "../helpers";
import { Input } from "@/components/atoms/Input/Input";

export const TextInputWidget = (props: WidgetProps) => {
  const { schema } = props;
  const mapped = mapJsonSchemaTypeToInputType(schema);

  const isTextArea = mapped === InputType.TEXT_AREA;
  const isNumber = mapped === InputType.NUMBER;
  const isInteger = mapped === InputType.INTEGER;
  const isPassword = mapped === InputType.PASSWORD;

  const htmlType = isTextArea
    ? "textarea"
    : isPassword
      ? "password"
      : isNumber
        ? "number"
        : "text";

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    const v = e.target.value;
    if (v === "") return props.onChange(undefined);
    if (isInteger || isNumber) return props.onChange(Number(v));
    return props.onChange(v);
  };

  return (
    <Input
      id={props.id}
      hideLabel={true}
      type={htmlType as any}
      label={""}
      size="small"
      wrapperClassName="mb-0"
      value={props.value ?? ""}
      onChange={handleChange as any}
      placeholder={schema.placeholder || ""}
      required={props.required}
      disabled={props.disabled}
    />
  );
};
