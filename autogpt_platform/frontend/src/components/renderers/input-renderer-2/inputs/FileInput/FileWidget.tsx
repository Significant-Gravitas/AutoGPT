import {
  RJSFSchema,
  FormContextType,
  StrictRJSFSchema,
  WidgetProps,
} from "@rjsf/utils";
import { FileInput } from "@/components/atoms/FileInput/FileInput";
type CustomWidgetProps<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
> = WidgetProps<T, S, F> & {
  options: any;
};
export const FileWidget = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(
  props: CustomWidgetProps<T, S, F>,
) => {
  const { onChange, disabled, readonly, value, schema, formContext } = props;

  const { size } = formContext || {};

  const displayName = schema?.title || "File";

  const handleChange = (fileUri: string) => {
    onChange(fileUri);
  };

  return (
    <FileInput
      variant={size === "large" ? "default" : "compact"}
      mode="base64"
      value={value}
      placeholder={displayName}
      onChange={handleChange}
      className={
        disabled || readonly ? "pointer-events-none opacity-50" : undefined
      }
    />
  );
};
