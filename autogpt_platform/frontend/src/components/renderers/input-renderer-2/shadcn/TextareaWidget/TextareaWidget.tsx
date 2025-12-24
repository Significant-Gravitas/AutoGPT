import { ChangeEvent, FocusEvent } from "react";
import {
  ariaDescribedByIds,
  FormContextType,
  RJSFSchema,
  StrictRJSFSchema,
  WidgetProps,
} from "@rjsf/utils";

import { Textarea } from "../components/ui/textarea";

type CustomWidgetProps<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
> = WidgetProps<T, S, F> & {
  options: any;
};

export default function TextareaWidget<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
  id,
  htmlName,
  placeholder,
  value,
  required,
  disabled,
  autofocus,
  readonly,
  onBlur,
  onFocus,
  onChange,
  options,
  className,
}: CustomWidgetProps<T, S, F>) {
  const _onChange = ({ target: { value } }: ChangeEvent<HTMLTextAreaElement>) =>
    onChange(value === "" ? options.emptyValue : value);
  const _onBlur = ({ target }: FocusEvent<HTMLTextAreaElement>) =>
    onBlur(id, target && target.value);
  const _onFocus = ({ target }: FocusEvent<HTMLTextAreaElement>) =>
    onFocus(id, target && target.value);

  return (
    <div className="flex p-0.5">
      <Textarea
        id={id}
        name={htmlName || id}
        placeholder={placeholder}
        disabled={disabled}
        readOnly={readonly}
        value={value ?? ""}
        required={required}
        autoFocus={autofocus}
        rows={options.rows || 5}
        onChange={_onChange}
        onBlur={_onBlur}
        onFocus={_onFocus}
        aria-describedby={ariaDescribedByIds(id)}
        className={className}
      />
    </div>
  );
}
