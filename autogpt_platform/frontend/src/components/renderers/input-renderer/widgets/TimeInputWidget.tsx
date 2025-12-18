import { WidgetProps } from "@rjsf/utils";
import { TimeInput } from "@/components/atoms/TimeInput/TimeInput";

export const TimeInputWidget = (props: WidgetProps) => {
  const { value, onChange, disabled, readonly, placeholder, id, formContext } =
    props;
  const { size = "small" } = formContext || {};

  // Determine input size based on context
  const inputSize = size === "large" ? "medium" : "small";

  return (
    <TimeInput
      value={value}
      onChange={onChange}
      className="w-full"
      label={""}
      id={id}
      hideLabel={true}
      size={inputSize as any}
      wrapperClassName="!mb-0 "
      disabled={disabled || readonly}
      placeholder={placeholder}
    />
  );
};
