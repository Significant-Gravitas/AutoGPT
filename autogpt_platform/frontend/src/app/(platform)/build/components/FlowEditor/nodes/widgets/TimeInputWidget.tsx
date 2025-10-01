import { WidgetProps } from "@rjsf/utils";
import { TimeInput } from "@/components/atoms/TimeInput/TimeInput";

export const TimeInputWidget = (props: WidgetProps) => {
  const { value, onChange, disabled, readonly, placeholder, id } = props;
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
      disabled={disabled || readonly}
      placeholder={placeholder}
    />
  );
};
