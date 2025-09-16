import { WidgetProps } from "@rjsf/utils";
import { InputRenderer, InputType } from "../InputRenderer";

export const TimeInputWidget = (props: WidgetProps) => {
  const { value, onChange, disabled, readonly, placeholder, autofocus } = props;
  return (
    <InputRenderer
      type={InputType.TIME}
      id={props.id}
      value={value}
      onChange={onChange}
      disabled={disabled}
      readonly={readonly}
      placeholder={placeholder || ""}
      autofocus={autofocus}
    />
  );
};
