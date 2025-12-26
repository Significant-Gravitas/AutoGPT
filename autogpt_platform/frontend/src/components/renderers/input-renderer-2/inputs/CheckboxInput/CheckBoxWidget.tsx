import {
  FormContextType,
  RJSFSchema,
  StrictRJSFSchema,
  WidgetProps,
} from "@rjsf/utils";
import { ExtendedFormContextType } from "../../../types";
import { Switch } from "@/components/atoms/Switch/Switch";

export function CheckboxWidget<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = ExtendedFormContextType,
>(props: WidgetProps<T, S, F>) {
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
