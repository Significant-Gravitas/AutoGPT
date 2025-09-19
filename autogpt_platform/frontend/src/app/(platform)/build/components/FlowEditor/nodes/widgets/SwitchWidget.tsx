import { WidgetProps } from "@rjsf/utils";
import { InputRenderer, InputType } from "../InputRenderer";
import { mapJsonSchemaTypeToInputType } from "../helpers";

export function SwitchWidget(props: WidgetProps) {
  const { value = false, onChange, disabled, readonly, autofocus } = props;

  const type = mapJsonSchemaTypeToInputType(props.schema);
  return (
    <InputRenderer
      type={type}
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
