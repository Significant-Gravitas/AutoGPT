import {
  FormNameContext,
  useIsStreaming,
  useFormName,
  useGetFieldValue,
  useSetDefaultValue,
  useStateField,
  useTriggerAction,
  type ComponentRenderProps,
} from "@openuidev/react-lang";
import { useId, type FormEvent } from "react";
import type { z } from "zod/v4";
import type { Field, Form } from "@/lib/openui/catalog";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { useOpenUIDisabled } from "../interactionContext";

export function FieldView({
  props,
}: ComponentRenderProps<z.infer<typeof Field.props>>) {
  const id = useId();
  const disabled = useOpenUIDisabled();
  const formName = useFormName();
  const getFieldValue = useGetFieldValue();
  useSetDefaultValue({
    formName,
    componentType: "Field",
    name: props.name,
    existingValue: getFieldValue(formName, props.name),
    defaultValue: props.value ?? "",
  });
  const field = useStateField(props.name, props.value);
  return (
    <Input
      id={id}
      label={props.label}
      labelVariant="body-medium"
      value={
        typeof field.value === "string" ? field.value : (props.value ?? "")
      }
      placeholder={props.placeholder}
      onChange={(event) => field.setValue(event.target.value)}
      maxLength={500}
      disabled={disabled}
      required
    />
  );
}

export function FormView({
  props,
  renderNode,
}: ComponentRenderProps<z.infer<typeof Form.props>>) {
  const triggerAction = useTriggerAction();
  const isStreaming = useIsStreaming();
  const disabled = useOpenUIDisabled();
  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!isStreaming && !disabled) triggerAction(props.message, props.name);
  }
  return (
    <FormNameContext.Provider value={props.name}>
      <form
        onSubmit={handleSubmit}
        className="space-y-5 rounded-xl border border-zinc-200 bg-white p-5"
      >
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
