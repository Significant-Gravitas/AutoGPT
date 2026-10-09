import { useId } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { NumberField } from "@/lib/openui/catalog-fields";
import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";
import { useOpenUIDisabled } from "../interactionContext";
import { useFieldView } from "./useFieldView";
import { useFieldValidation } from "./useFieldValidation";
import { numberFieldError } from "./fieldValidation";
import { FieldFeedback } from "./FieldFeedback";

export function NumberFieldView({
  props,
}: ComponentRenderProps<z.infer<typeof NumberField.props>>) {
  const id = useId();
  const disabled = useOpenUIDisabled();
  const field = useFieldView<string | number>(props.name, props.value ?? "");
  const validation = useFieldValidation(
    numberFieldError(String(field.value), props),
  );
  const { error, ...handlers } = validation;
  return (
    <FieldFeedback id={`${id}-error`} error={error}>
      <Input
        {...handlers}
        id={id}
        name={props.name}
        label={props.label}
        labelVariant="body-medium"
        type="text"
        inputMode="decimal"
        data-openui-number
        value={field.value}
        maxLength={100}
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
