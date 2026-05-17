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

  return (
    <Switch
      id={id}
      checked={Boolean(value)}
      onCheckedChange={(checked) => onChange(checked)}
      disabled={disabled || readonly}
      autoFocus={autofocus}
      {...(accessibleLabel ? { "aria-label": accessibleLabel } : {})}
    />
  );
}
