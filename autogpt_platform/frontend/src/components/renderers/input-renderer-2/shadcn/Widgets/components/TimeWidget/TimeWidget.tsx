import {
  RJSFSchema,
  StrictRJSFSchema,
  FormContextType,
  WidgetProps,
} from "@rjsf/utils";
import { TimeInput } from "@/components/atoms/TimeInput/TimeInput";

type CustomWidgetProps<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
> = WidgetProps<T, S, F> & {
  options: any;
};
export const TimeWidget = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(
  props: CustomWidgetProps<T, S, F>,
) => {
  const { value, onChange, disabled, readonly, placeholder, id, formContext } =
    props;
  const { size = "small" } = formContext || {};

  // Determine input size based on context
  const inputSize = size === "large" ? "medium" : "small";

  return (
    <TimeInput
      value={value}
      onChange={onChange}
      className="w-full"
      label={""}
      id={id}
      hideLabel={true}
      size={inputSize as any}
      wrapperClassName="!mb-0 "
      disabled={disabled || readonly}
      placeholder={placeholder}
    />
  );
};
