import { useId } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type {
  TextAreaField,
  ToggleField,
  MultiSelectField,
} from "@/lib/openui/catalog-rich-fields";
import { Input } from "@/components/atoms/Input/Input";
import { useOpenUIDisabled } from "../interactionContext";
import { useFieldView } from "./useFieldView";
import { useFieldValidation } from "./useFieldValidation";
import { FieldFeedback } from "./FieldFeedback";

export function TextAreaFieldView({
  props,
}: ComponentRenderProps<z.infer<typeof TextAreaField.props>>) {
  const id = useId();
  const disabled = useOpenUIDisabled();
  const field = useFieldView(props.name, props.value ?? "");
  const { error, ...handlers } = useFieldValidation(
    props.required && !String(field.value).trim() ? "Fill in this field." : "",
  );
  return (
    <FieldFeedback id={`${id}-error`} error={error}>
      <Input
        {...handlers}
        id={id}
        type="textarea"
        rows={3}
        label={props.label}
        labelVariant="body-medium"
        value={field.value}
        placeholder={props.placeholder}
        maxLength={2000}
        required={props.required}
        disabled={disabled}
        onChange={(event) => field.setValue(event.target.value)}
        aria-invalid={!!error}
        aria-describedby={`${id}-error`}
      />
    </FieldFeedback>
  );
}

export function ToggleFieldView({
  props,
}: ComponentRenderProps<z.infer<typeof ToggleField.props>>) {
  const disabled = useOpenUIDisabled();
  const field = useFieldView(props.name, props.value ?? false);
  return (
    <label className="flex min-h-11 cursor-pointer items-center gap-3 text-sm text-foreground">
      <input
        type="checkbox"
        className="h-5 w-5 accent-primary"
        checked={field.value === true}
        disabled={disabled}
        onChange={(event) => field.setValue(event.target.checked)}
      />
      {props.label}
    </label>
  );
}

export function MultiSelectFieldView({
  props,
}: ComponentRenderProps<z.infer<typeof MultiSelectField.props>>) {
  const id = useId();
  const disabled = useOpenUIDisabled();
  const field = useFieldView(props.name, props.value ?? []);
  const selected = Array.isArray(field.value)
    ? field.value.filter((value) =>
        props.options?.some((option) => option.value === value),
      )
    : [];
  const { error, ...handlers } = useFieldValidation(
    props.required && !selected.length ? "Choose at least one option." : "",
  );
  return (
    <fieldset
      disabled={disabled}
      className="space-y-2"
      aria-describedby={`${id}-error`}
    >
      <legend className="text-sm font-medium text-foreground">
        {props.label}
      </legend>
      <div className="flex flex-wrap gap-2">
        {props.options?.map((option, index) => (
          <label
            key={option.value}
            className="flex min-h-11 cursor-pointer items-center gap-2 rounded-lg border border-border px-3 text-sm has-[:checked]:border-primary has-[:checked]:bg-primary/5"
          >
            <input
              {...(index === 0 ? handlers : {})}
              type="checkbox"
              className="h-4 w-4 accent-primary"
              checked={selected.includes(option.value)}
              aria-invalid={!!error}
              onChange={(event) =>
                field.setValue(
                  event.target.checked
                    ? [...selected, option.value]
                    : selected.filter((value) => value !== option.value),
                )
              }
            />
            {option.label}
          </label>
        ))}
      </div>
      <p
        id={`${id}-error`}
        aria-live="polite"
        className="text-xs text-destructive"
      >
        {error}
      </p>
    </fieldset>
  );
}
