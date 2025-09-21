import * as React from "react";
import { WidgetProps } from "@rjsf/utils";
import { DateInput } from "@/components/atoms/DateInput/DateInput";

export const DateInputWidget = (props: WidgetProps) => {
  const { value, onChange, disabled, readonly, placeholder, autofocus, id } =
    props;

  return (
    <DateInput
      size="small"
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
