import {
  FormContextType,
  RJSFSchema,
  StrictRJSFSchema,
  WidgetProps,
} from "@rjsf/utils";
import { ExtendedFormContextType } from "../../../../types";
import { Select } from "@/components/atoms/Select/Select";
import { isOptionalType } from "@/components/renderers/input-renderer-2/utils/schema-utils";

type CustomWidgetProps<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = ExtendedFormContextType,
> = WidgetProps<T, S, F> & {
  options: any;
};

export const AnyOfSelector = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = ExtendedFormContextType,
>(
  props: CustomWidgetProps<T, S, F>,
) => {
  const { id, disabled, readonly, value, onChange, options, schema } = props;
  const enumOptions = options.enumOptions || [];

  if (isOptionalType(schema)) return <div>Oh yes</div>;
  return (
    <div>
      <Select
        label=""
        id={id}
        hideLabel={true}
        disabled={disabled || readonly}
        size={"small"}
        value={value ?? ""}
        onValueChange={onChange}
        options={
          enumOptions?.map((option: any) => ({
            value: option.value,
            label: option.label,
          })) || []
        }
        wrapperClassName="!mb-0 "
        className="h-6 w-fit gap-1 pl-3 pr-2"
      />
    </div>
  );
};
