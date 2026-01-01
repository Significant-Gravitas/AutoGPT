import { FieldProps, getUiOptions, getWidget } from "@rjsf/utils";
import { AnyOfFieldTitle } from "./components/AnyOfFieldTitle";
import { isEmpty } from "lodash";
import { useAnyOfField } from "./useAnyOfField";
import { getHandleId, updateUiOption } from "../../helpers";
import { ANY_OF_FLAG } from "../../constants";

export const AnyOfField = (props: FieldProps) => {
  const { registry, schema } = props;
  const { fields } = registry;
  const { SchemaField: _SchemaField } = fields;

  const uiOptions = getUiOptions(props.uiSchema, props.globalUiOptions);

  const Widget = getWidget({ type: "string" }, "select", registry.widgets);

  const {
    handleOptionChange,
    enumOptions,
    selectedOption,
    optionSchema,
    field_id,
  } = useAnyOfField(props);

  const handleId = getHandleId({
    uiOptions,
    id: field_id,
    schema: schema,
  });
  const updatedUiSchema = updateUiOption(props.uiSchema, {
    handleId: handleId + ANY_OF_FLAG,
    label: false,
    fromAnyOf: true,
  });

  const optionsSchemaField =
    (optionSchema && optionSchema.type !== "null" && (
      <_SchemaField
        {...props}
        schema={optionSchema}
        uiSchema={updatedUiSchema}
      />
    )) ||
    null;

  const selector = (
    <Widget
      id={field_id}
      name={`${props.name}${schema.oneOf ? "__oneof_select" : "__anyof_select"}`}
      schema={{ type: "number", default: 0 }}
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
      <AnyOfFieldTitle
        {...props}
        selector={selector}
        uiSchema={updatedUiSchema}
      />
      {optionsSchemaField}
    </div>
  );
};
