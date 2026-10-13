import { useId } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { RangeField } from "@/lib/openui/catalog-inputs";
import { Input } from "@/components/atoms/Input/Input";
import { asNumber } from "@/lib/openui/calculations";
import { numberFieldError } from "../fieldValidation";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";
import { useOpenUIDisabled } from "../../interactionContext";
import { ControlValidity } from "../ControlValidity";

export function RangeFieldView({
  props,
}: ComponentRenderProps<z.infer<typeof RangeField.props>>) {
  const id = useId();
  const field = useFieldView<{
    low: string | number | null;
    high: string | number | null;
  }>(`${props.name}__draft`, props.value);
  const disabled = useOpenUIDisabled();
  const low = asNumber(field.value.low),
    high = asNumber(field.value.high);
  const error =
    numberFieldError(String(field.value.low ?? ""), {
      min: props.min,
      max: props.max,
      step: props.step,
    }) ||
    numberFieldError(String(field.value.high ?? ""), {
      min: props.min,
      max: props.max,
      step: props.step,
    }) ||
    (low != null && high != null && low > high
      ? "Lower bound must not exceed the upper bound."
      : "");
  useDerivedField(props.name, { low, high });
  useDerivedField(`${props.name}_low`, error ? null : low);
  useDerivedField(`${props.name}_high`, error ? null : high);
  useDerivedField(`${props.name}__valid`, !error);
  return (
    <fieldset disabled={disabled} className="space-y-2">
      <legend className="text-sm font-semibold">
        {props.label} ({props.unit})
      </legend>
      <div className="grid gap-3 sm:grid-cols-2">
        {(["low", "high"] as const).map((bound) => (
          <div key={bound} className="space-y-2">
            <Input
              id={`${id}-${bound}`}
              label={`${props.label} ${bound}`}
              value={String(field.value[bound] ?? "")}
              inputMode="decimal"
              onChange={(event) =>
                field.setValue({ ...field.value, [bound]: event.target.value })
              }
            />
            <input
              aria-label={`${props.label} ${bound} slider`}
              type="range"
              min={props.min}
              max={props.max}
              step={props.step}
              disabled={disabled || asNumber(field.value[bound]) === null}
              value={asNumber(field.value[bound]) ?? props.min}
              onChange={(event) =>
                field.setValue({
                  ...field.value,
                  [bound]: Number(event.target.value),
                })
              }
              className="min-h-11 w-full accent-primary"
            />
          </div>
        ))}
      </div>
      <ControlValidity label={props.label} error={error} />
    </fieldset>
  );
}
