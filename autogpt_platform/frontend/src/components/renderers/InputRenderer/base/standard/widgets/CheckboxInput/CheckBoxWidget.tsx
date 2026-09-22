import { useFieldAccessibility } from "../../../../field-accessibility";
import { WidgetProps } from "@rjsf/utils";
import { Switch } from "@/components/atoms/Switch/Switch";

export function CheckboxWidget(props: WidgetProps) {
  const {
    value = false,
    onChange,
    disabled,
    readonly,
    autofocus,
    id,
    schema,
    label,
  } = props;
  const accessibleLabel = schema.title || label;
  const accessibility = useFieldAccessibility(
    id,
    accessibleLabel,
    props.formContext ?? props.registry?.formContext,
    props["aria-describedby"],
  );

  return (
    <Switch
      {...accessibility}
      checked={Boolean(value)}
      onCheckedChange={(checked) => onChange(checked)}
      disabled={disabled || readonly}
      autoFocus={autofocus}
    />
  );
}
