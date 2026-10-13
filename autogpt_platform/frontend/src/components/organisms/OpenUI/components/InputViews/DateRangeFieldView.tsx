import { useId } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import { z } from "zod/v4";
import { DateRangeField } from "@/lib/openui/catalog-inputs";
import { Input } from "@/components/atoms/Input/Input";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";
import { useOpenUIDisabled } from "../../interactionContext";
import { ControlValidity } from "../ControlValidity";

export function dateRangeError(
  value: { start: string; end: string },
  required: boolean,
) {
  if (!required && !value.start && !value.end) return "";
  if (
    !z.iso.date().safeParse(value.start).success ||
    !z.iso.date().safeParse(value.end).success
  )
    return "Choose both valid dates.";
  return value.end < value.start
    ? "End date must be on or after the start date."
    : "";
}

export function DateRangeFieldView({
  props,
}: ComponentRenderProps<z.infer<typeof DateRangeField.props>>) {
  const id = useId();
  const field = useFieldView(props.name, props.value);
  const parsed = DateRangeField.props.shape.value.safeParse(field.value);
  const value = parsed.success ? parsed.data : props.value;
  const disabled = useOpenUIDisabled();
  const error = dateRangeError(value, props.required);
  useDerivedField(`${props.name}__valid`, !error);
  return (
    <fieldset disabled={disabled} className="space-y-2">
      <legend className="text-sm font-semibold">{props.label}</legend>
      <div className="grid gap-3 sm:grid-cols-2">
        <Input
          id={`${id}-start`}
          label={`${props.label} start`}
          type="date"
          value={value.start}
          required={props.required}
          onChange={(event) =>
            field.setValue({ ...value, start: event.target.value })
          }
        />
        <Input
          id={`${id}-end`}
          label={`${props.label} end`}
          type="date"
          value={value.end}
          required={props.required}
          onChange={(event) =>
            field.setValue({ ...value, end: event.target.value })
          }
        />
      </div>
      <ControlValidity label={props.label} error={error} />
    </fieldset>
  );
}
