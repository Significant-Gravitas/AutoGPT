import { useId } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import {
  localDateTime,
  type DateTimeField,
} from "@/lib/openui/catalog-temporal";
import { localInstants, resolveDateTime } from "@/lib/openui/temporal";
import { Input } from "@/components/atoms/Input/Input";
import { useOpenUIDisabled } from "../../interactionContext";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";
import { ControlValidity } from "../ControlValidity";

export function DateTimeFieldView({
  props,
}: ComponentRenderProps<z.infer<typeof DateTimeField.props>>) {
  const id = useId();
  const field = useFieldView(`${props.name}__draft`, props.value);
  const parsed = localDateTime.safeParse(field.value);
  const value = parsed.success ? parsed.data : props.value;
  const disabled = useOpenUIDisabled();
  const result = resolveDateTime(value, props.required);
  const repeated = localInstants(value.local, value.timeZone).length > 1;
  useDerivedField(props.name, { ...value, instant: result.instant });
  useDerivedField(`${props.name}__valid`, !result.error);
  return (
    <fieldset disabled={disabled} className="space-y-3">
      <legend className="text-sm font-medium">{props.label}</legend>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="space-y-1 text-xs">
          Local date and time
          <Input
            label=""
            id={`${id}-date`}
            aria-label={`${props.label} date and time`}
            type="datetime-local"
            value={value.local}
            onChange={(event) =>
              field.setValue({
                ...value,
                local: event.target.value,
                occurrence: "",
              })
            }
          />
        </label>
        <label className="space-y-1 text-xs">
          Time zone
          <Input
            label=""
            id={`${id}-zone`}
            aria-label={`${props.label} time zone`}
            value={value.timeZone}
            placeholder="America/Los_Angeles"
            maxLength={100}
            onChange={(event) =>
              field.setValue({
                ...value,
                timeZone: event.target.value,
                occurrence: "",
              })
            }
          />
        </label>
      </div>
      {repeated && (
        <label className="block space-y-1 text-sm">
          Clock change occurrence
          <select
            aria-label={`${props.label} occurrence`}
            className="min-h-11 w-full rounded-md border bg-background p-2"
            value={value.occurrence}
            onChange={(event) =>
              field.setValue({
                ...value,
                occurrence: event.target.value as typeof value.occurrence,
              })
            }
          >
            <option value="">Choose an occurrence</option>
            <option value="earlier">Earlier occurrence</option>
            <option value="later">Later occurrence</option>
          </select>
        </label>
      )}
      <ControlValidity label={props.label} error={result.error} />
      {result.instant && (
        <output className="block text-xs text-muted-foreground">
          UTC: {result.instant}
        </output>
      )}
    </fieldset>
  );
}
