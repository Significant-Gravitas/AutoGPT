import {
  FormNameContext,
  useIsStreaming,
  useTriggerAction,
  useSetFieldValue,
  type ComponentRenderProps,
} from "@openuidev/react-lang";
import { useId, type FormEvent } from "react";
import type { z } from "zod/v4";
import type { Field, Form } from "@/lib/openui/catalog-fields";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { useOpenUIDisabled } from "../interactionContext";
import { useFieldView } from "./useFieldView";
import { validateForm } from "./fieldValidation";
import { useFieldValidation } from "./useFieldValidation";
import { FieldFeedback } from "./FieldFeedback";
import { cn } from "@/lib/utils";

export function FieldView({
  props,
}: ComponentRenderProps<z.infer<typeof Field.props>>) {
  const id = useId();
  const disabled = useOpenUIDisabled();
  const field = useFieldView(props.name, props.value ?? "");
  const { error, ...handlers } = useFieldValidation(
    String(field.value).trim() ? "" : "Fill in this field.",
  );
  return (
    <FieldFeedback id={`${id}-error`} error={error}>
      <Input
        {...handlers}
        id={id}
        label={props.label}
        labelVariant="body-medium"
        value={
          typeof field.value === "string" ? field.value : (props.value ?? "")
        }
        placeholder={props.placeholder}
        onChange={(event) => field.setValue(event.target.value)}
        aria-invalid={!!error}
        aria-describedby={`${id}-error`}
        className={cn(
          error && "border-red-500 focus:border-red-500 focus:ring-red-500",
        )}
        maxLength={500}
        disabled={disabled}
        required
      />
    </FieldFeedback>
  );
}

export function FormView({
  props,
  renderNode,
}: ComponentRenderProps<z.infer<typeof Form.props>>) {
  const triggerAction = useTriggerAction();
  const isStreaming = useIsStreaming();
  const disabled = useOpenUIDisabled();
  const setFieldValue = useSetFieldValue();
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isStreaming || disabled || !validateForm(event.currentTarget)) return;
    for (const input of event.currentTarget.querySelectorAll<HTMLInputElement>(
      "input[data-openui-number]",
    )) {
      setFieldValue(props.name, "NumberField", input.name, Number(input.value));
    }
    triggerAction(props.message, props.name);
  }
  return (
    <FormNameContext.Provider value={props.name}>
      <form onSubmit={handleSubmit} noValidate className="space-y-5">
        <h3 className="text-sm font-semibold text-zinc-800">{props.title}</h3>
        {renderNode(props.fields?.slice(0, 6))}
        <Button
          type="submit"
          size="small"
          disabled={isStreaming || disabled}
          rightIcon={<Icon icon={ArrowRight01Icon} size={16} />}
          unmask={false}
        >
          {props.submitLabel}
        </Button>
      </form>
    </FormNameContext.Provider>
  );
}
