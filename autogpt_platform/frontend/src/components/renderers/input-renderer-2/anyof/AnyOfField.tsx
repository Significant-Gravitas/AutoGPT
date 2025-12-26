import {
  FieldProps,
  FormContextType,
  getUiOptions,
  getWidget,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import { AnyOfFieldTitle } from "./components/AnyOfFieldTitle";
import { isEmpty } from "lodash";
import { useAnyOfField } from "./useAnyOfField";

export const AnyOfField = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(
  props: FieldProps<T, S, F>,
) => {
  const { registry, schema } = props;
  const { fields } = registry;
  const { SchemaField: _SchemaField } = fields;

  const { widget = "select" } = getUiOptions<T, S, F>(
    props.uiSchema,
    props.globalUiOptions,
  );

  const Widget = getWidget<T, S, F>(
    { type: "string" },
    widget,
    props.registry.widgets,
  );

  const {
    handleOptionChange,
    enumOptions,
    selectedOption,
    optionSchema,
    field_id,
  } = useAnyOfField(props);

  const optionsSchemaField =
    (optionSchema && optionSchema.type !== "null" && (
      <_SchemaField
        {...props}
        schema={optionSchema}
        uiSchema={{
          ...props.uiSchema,
          "ui:options": {
            ...props.uiSchema?.["ui:options"],
            label: false,
          },
        }}
      />
    )) ||
    null;

  const selector = (
    <Widget
      id={field_id}
      name={`${props.name}${schema.oneOf ? "__oneof_select" : "__anyof_select"}`}
      schema={{ type: "number", default: 0 } as S}
      onChange={handleOptionChange}
      onBlur={props.onBlur}
      onFocus={props.onFocus}
      disabled={props.disabled || isEmpty(enumOptions)}
      multiple={false}
      value={selectedOption >= 0 ? selectedOption : undefined}
      options={{ enumOptions }}
      registry={registry}
      placeholder={props.placeholder}
      autocomplete={props.autocomplete}
      className="h-[22px] w-fit gap-1 border-none bg-zinc-100 px-1 pl-3 text-xs font-medium"
      autofocus={props.autofocus}
      label=""
      hideLabel={true}
      readonly={props.readonly}
    />
  );

  return (
    <div>
      <AnyOfFieldTitle {...props} selector={selector} />
      {optionsSchemaField}
    </div>
  );
};
