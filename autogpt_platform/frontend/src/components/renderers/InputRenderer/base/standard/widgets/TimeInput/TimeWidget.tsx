import { useFieldAccessibility } from "../../../../field-accessibility";
import { WidgetProps } from "@rjsf/utils";
import { TimeInput } from "@/components/atoms/TimeInput/TimeInput";

export const TimeWidget = (props: WidgetProps) => {
  const {
    value,
    onChange,
    disabled,
    readonly,
    placeholder,
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
    <TimeInput
      value={value}
      onChange={onChange}
      className="w-full"
      label={schema.title || label || ""}
      {...accessibility}
      hideLabel={true}
      size={inputSize as any}
      wrapperClassName="!mb-0 "
      disabled={disabled || readonly}
      placeholder={placeholder}
    />
  );
};
