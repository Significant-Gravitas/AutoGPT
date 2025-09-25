import { WidgetProps } from "@rjsf/utils";
import { DateTimeInput } from "@/components/atoms/DateTimeInput/DateTimeInput";

export const DateTimeInputWidget = (props: WidgetProps) => {
  const { value, onChange, disabled, readonly, placeholder, autofocus, id } =
    props;
  return (
    <DateTimeInput
      size="small"
      id={id}
      hideLabel={true}
      label={""}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      disabled={disabled}
      readonly={readonly}
      autoFocus={autofocus}
    />
  );
};
