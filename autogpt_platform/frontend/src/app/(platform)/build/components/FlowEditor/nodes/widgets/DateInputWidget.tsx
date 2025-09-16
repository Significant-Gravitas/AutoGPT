import * as React from "react";
import { WidgetProps } from "@rjsf/utils";
import { InputRenderer, InputType } from "../InputRenderer";

export const DateInputWidget = (props: WidgetProps) => {
  const { value, onChange, disabled, readonly, placeholder, autofocus } = props;

  return (
    <InputRenderer
      type={InputType.DATE}
      value={value}
      id={props.id}
      placeholder={placeholder || ""}
      required={props.required}
      onChange={onChange}
      disabled={disabled}
      readonly={readonly}
      autofocus={autofocus}
    />
  );
};
