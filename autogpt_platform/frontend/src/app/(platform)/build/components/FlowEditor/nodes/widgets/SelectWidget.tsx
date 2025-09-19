import { WidgetProps } from "@rjsf/utils";
import { InputRenderer } from "../InputRenderer";
import { mapJsonSchemaTypeToInputType } from "../helpers";

export const SelectWidget = (props: WidgetProps) => {
  const { options, value, onChange, disabled, readonly, multiple } = props;
  const enumOptions = options.enumOptions || [];
  const type = mapJsonSchemaTypeToInputType(props.schema);
  return (
    <InputRenderer
      type={type}
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
