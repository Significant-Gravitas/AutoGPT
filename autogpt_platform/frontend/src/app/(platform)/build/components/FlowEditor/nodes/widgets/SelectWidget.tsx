import { WidgetProps } from "@rjsf/utils";
import { InputRenderer, InputType } from "../InputRenderer";

export const SelectWidget = (props: WidgetProps) => {
  const { options, value, onChange, disabled, readonly, multiple } = props;
  const enumOptions = options.enumOptions || [];

  return (
    <InputRenderer
      type={InputType.SELECT}
      value={value}
      id={props.id}
      placeholder={props.placeholder || ""}
      required={props.required}
      disabled={disabled}
      readonly={readonly}
      autofocus={props.autofocus}
      options={enumOptions}
      onChange={onChange}
      multiple={multiple}
    />
  );
};
