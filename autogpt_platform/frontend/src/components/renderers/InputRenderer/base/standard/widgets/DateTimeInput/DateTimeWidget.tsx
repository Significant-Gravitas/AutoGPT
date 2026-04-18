import { WidgetProps } from "@rjsf/utils";
import { DateTimeInput } from "@/components/atoms/DateTimeInput/DateTimeInput";

export const DateTimeWidget = (props: WidgetProps) => {
  const {
    value,
    onChange,
    disabled,
    readonly,
    placeholder,
    autofocus,
    id,
    formContext,
    schema,
    label,
  } = props;
  const { size = "small" } = formContext || {};

  // Determine input size based on context
  const inputSize = size === "large" ? "medium" : "small";

  return (
    <DateTimeInput
      size={inputSize as any}
      id={id}
      hideLabel={true}
      label={schema.title || label || ""}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      disabled={disabled}
      readonly={readonly}
      autoFocus={autofocus}
    />
  );
};
