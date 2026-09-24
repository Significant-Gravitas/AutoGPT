import { useFieldAccessibility } from "../../../../field-accessibility";
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
  const accessibility = useFieldAccessibility(
    id,
    schema.title || label,
    formContext ?? props.registry?.formContext,
    props["aria-describedby"],
  );

  // Determine input size based on context
  const inputSize = size === "large" ? "medium" : "small";

  return (
    <DateTimeInput
      size={inputSize as any}
      {...accessibility}
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
