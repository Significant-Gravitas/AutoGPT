import { WidgetProps } from "@rjsf/utils";
import { InputRenderer, InputType } from "../InputRenderer";
import { mapJsonSchemaTypeToInputType } from "../helpers";

export const TextInputWidget = (props: WidgetProps) => {
  const { schema } = props;

  const type = mapJsonSchemaTypeToInputType(schema);

  return (
    <InputRenderer
      type={type}
      value={props.value}
      id={props.id}
      placeholder={schema.placeholder || ""}
      required={props.required}
      onChange={props.onChange}
      disabled={props.disabled}
    />
  );
};
