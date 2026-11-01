import { getDefaultRegistry } from "@rjsf/core";
import { FieldProps } from "@rjsf/utils";

const { NumberField: RjsfNumberField } = getDefaultRegistry().fields;

/**
 * RJSF's NumberField turns the widget's text into a JS number, which rounds
 * integers beyond 2^53. Integer fields skip it: their widget already parses
 * the text and keeps such an integer as its digit string.
 */
export function NumberField(props: FieldProps) {
  if (props.schema.type !== "integer") return <RjsfNumberField {...props} />;
  const { StringField } = props.registry.fields;
  return <StringField {...props} />;
}
