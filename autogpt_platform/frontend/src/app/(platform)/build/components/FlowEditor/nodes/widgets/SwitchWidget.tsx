import { WidgetProps } from "@rjsf/utils";
import { InputRenderer, InputType } from "../InputRenderer";

export function SwitchWidget(props: WidgetProps) {
  const { value = false, onChange, disabled, readonly, autofocus } = props;

  return (
    <InputRenderer
      type={InputType.BOOLEAN}
      onChange={onChange}
      value={value}
      id={props.id}
      placeholder={props.placeholder || ""}
      required={props.required}
      disabled={disabled}
      readonly={readonly}
      autofocus={autofocus}
    />
  );
}
