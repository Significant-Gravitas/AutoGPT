import * as React from "react";
import { WidgetProps } from "@rjsf/utils";
import { DateInput } from "@/components/atoms/DateInput/DateInput";

export const DateInputWidget = (props: WidgetProps) => {
  const {
    value,
    onChange,
    disabled,
    readonly,
    placeholder,
    autofocus,
    id,
    formContext,
  } = props;
  const { size = "small" } = formContext || {};

  // Determine input size based on context
  const inputSize = size === "large" ? "default" : "small";

  return (
    <DateInput
      size={inputSize as any}
      id={id}
      hideLabel={true}
      label={""}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      disabled={disabled}
      readonly={readonly}
      autoFocus={autofocus}
    />
  );
};
