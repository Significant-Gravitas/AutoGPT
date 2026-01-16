import { FieldProps, getUiOptions, getWidget } from "@rjsf/utils";
import { AnyOfFieldTitle } from "./components/AnyOfFieldTitle";
import { isEmpty } from "lodash";
import { useAnyOfField } from "./useAnyOfField";
import { cleanUpHandleId, getHandleId, updateUiOption } from "../../helpers";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { ANY_OF_FLAG } from "../../constants";
import { findCustomFieldId } from "../../registry";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { cn } from "@/lib/utils";

export const AnyOfField = (props: FieldProps) => {
  const { registry, schema } = props;
  const { fields } = registry;
  const { SchemaField: _SchemaField } = fields;
  const { nodeId } = registry.formContext;
  const { isInputConnected } = useEdgeStore();
  const {
    handleOptionChange,
    enumOptions,
    selectedOption,
    optionSchema,
    field_id,
  } = useAnyOfField(props);

  const isInputBroken = useNodeStore((state) => state.isInputBroken);

  const parentCustomFieldId = findCustomFieldId(schema);
  if (parentCustomFieldId) {
    return null;
  }

  const uiOptions = getUiOptions(props.uiSchema, props.globalUiOptions);

  const Widget = getWidget({ type: "string" }, "select", registry.widgets);

  const handleId = getHandleId({
    uiOptions,
    id: field_id + ANY_OF_FLAG,
    schema: schema,
  });

  const updatedUiSchema = updateUiOption(props.uiSchema, {
    handleId: handleId,
    label: false,
    fromAnyOf: true,
  });

  const isHandleConnected = isInputConnected(nodeId, handleId);
  const isAnyOfInputBroken = isInputBroken(nodeId, cleanUpHandleId(handleId));

  // Now anyOf can render - custom fields if the option schema matches a custom field
  const optionCustomFieldId = optionSchema
    ? findCustomFieldId(optionSchema)
    : null;

  const optionUiSchema = optionCustomFieldId
    ? { ...updatedUiSchema, "ui:field": optionCustomFieldId }
    : updatedUiSchema;

  const optionsSchemaField =
    (optionSchema && optionSchema.type !== "null" && (
      <_SchemaField
        {...props}
        schema={optionSchema}
        uiSchema={optionUiSchema}
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
      className={cn(
        "-ml-1 h-[22px] w-fit gap-1 px-1 pl-2 text-xs font-medium",
        isAnyOfInputBroken &&
          "border-red-500 bg-red-100 text-red-600 line-through",
      )}
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
      {!isHandleConnected && !isAnyOfInputBroken && optionsSchemaField}
    </div>
  );
};
