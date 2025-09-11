import { Switch } from "@/components/atoms/Switch/Switch";
import { WidgetProps } from "@rjsf/utils";

export function SwitchWidget(props: WidgetProps) {
  const { value = false, onChange, disabled, readonly, autofocus } = props;

  return (
    <Switch
      checked={Boolean(value)}
      onCheckedChange={(checked) => onChange(checked)}
      disabled={disabled || readonly}
      autoFocus={autofocus}
    />
  );
}
