import { WidgetProps } from "@rjsf/utils";
import { InputRenderer, InputType } from "../InputRenderer";

export const DateTimeInputWidget = (props: WidgetProps) => {
  const { value, onChange, disabled, readonly, placeholder, autofocus } = props;

  return (
    <InputRenderer
      value={value}
      onChange={onChange}
      type={InputType.DATE_TIME}
      id={props.id}
      placeholder={placeholder || ""}
      required={props.required}
      disabled={disabled}
      readonly={readonly}
      autofocus={autofocus}
    />
  );
};
