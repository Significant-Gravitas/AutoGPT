import { WidgetProps } from "@rjsf/utils";
import { InputRenderer, InputType } from "../InputRenderer";

export const TextInputWidget = (props: WidgetProps) => {
  return (
    <InputRenderer
      type={InputType.STRING}
      value={props.value}
      id={props.id}
      placeholder={props.placeholder || ""}
      required={props.required}
      onChange={props.onChange}
      disabled={props.disabled}
      readonly={props.readonly}
      autofocus={props.autofocus}
    />
  );
};
