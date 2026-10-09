import { useId } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { SelectField, DateField } from "@/lib/openui/catalog-fields";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { useOpenUIDisabled } from "../interactionContext";
import { useFieldView } from "./useFieldView";
import { useFieldValidation } from "./useFieldValidation";
import { FieldFeedback } from "./FieldFeedback";
import { cn } from "@/lib/utils";

export function SelectFieldView({
  props,
}: ComponentRenderProps<z.infer<typeof SelectField.props>>) {
  const id = useId();
  const disabled = useOpenUIDisabled();
  const options = (props.options ?? [])
    .filter(
      (option, index, all) =>
        option?.value &&
        all.findIndex((item) => item?.value === option.value) === index,
    )
    .slice(0, 12);
  const initialValue = options.some((option) => option.value === props.value)
    ? props.value
    : (options[0]?.value ?? "");
  const field = useFieldView(props.name, initialValue);
  function selectValue(value: string) {
    if (options.some((option) => option.value === value)) field.setValue(value);
  }
  return (
    <Select
      id={id}
      label={props.label}
      labelVariant="body-medium"
      options={options}
      value={field.value}
      onValueChange={selectValue}
      disabled={disabled}
    />
  );
}

export function DateFieldView({
  props,
}: ComponentRenderProps<z.infer<typeof DateField.props>>) {
  const id = useId();
  const disabled = useOpenUIDisabled();
  const field = useFieldView(props.name, props.value ?? "");
  const { error, ...handlers } = useFieldValidation(
    field.value ? "" : "Choose a valid date.",
  );
  return (
    <FieldFeedback id={`${id}-error`} error={error}>
      <Input
        {...handlers}
        id={id}
        label={props.label}
        labelVariant="body-medium"
        type="date"
        value={field.value}
        onChange={(event) => field.setValue(event.target.value)}
        aria-invalid={!!error}
        aria-describedby={`${id}-error`}
        className={cn(
          error && "border-red-500 focus:border-red-500 focus:ring-red-500",
        )}
        disabled={disabled}
        required
      />
    </FieldFeedback>
  );
}
