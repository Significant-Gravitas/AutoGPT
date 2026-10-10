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
import { useDerivedField } from "./useDerivedField";

export function NumberFieldView({
  props,
}: ComponentRenderProps<z.infer<typeof NumberField.props>>) {
  const id = useId();
  const disabled = useOpenUIDisabled();
  const field = useFieldView<string | number>(props.name, props.value ?? "");
  const message = numberFieldError(String(field.value), props);
  useDerivedField(`${props.name}__valid`, !message);
  const validation = useFieldValidation(message, true);
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
