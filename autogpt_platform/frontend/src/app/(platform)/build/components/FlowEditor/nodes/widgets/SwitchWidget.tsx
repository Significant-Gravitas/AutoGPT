import { WidgetProps } from "@rjsf/utils";
import { Switch } from "@/components/atoms/Switch/Switch";

export function SwitchWidget(props: WidgetProps) {
  const { value = false, onChange, disabled, readonly, autofocus, id } = props;
  return (
    <Switch
      id={id}
      checked={Boolean(value)}
      onCheckedChange={(checked) => onChange(checked)}
      disabled={disabled || readonly}
      autoFocus={autofocus}
    />
  );
}
