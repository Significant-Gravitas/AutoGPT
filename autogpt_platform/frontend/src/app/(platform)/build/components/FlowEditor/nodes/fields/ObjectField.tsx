import React from "react";
import { FieldProps } from "@rjsf/utils";
import { getDefaultRegistry } from "@rjsf/core";
import { InputRenderer, InputType } from "../InputRenderer";
import { generateHandleId } from "../../handlers/helpers";
import { mapJsonSchemaTypeToInputType } from "../helpers";

export const ObjectField = (props: FieldProps) => {
  const {
    schema,
    formData = {},
    onChange,
    name,
    idSchema,
    formContext,
  } = props;
  const DefaultObjectField = getDefaultRegistry().fields.ObjectField;

  // Let the default field render for root or fixed-schema objects
  const isFreeForm =
    !schema.properties ||
    Object.keys(schema.properties).length === 0 ||
    schema.additionalProperties === true;

  if (idSchema?.$id === "root" || !isFreeForm) {
    return <DefaultObjectField {...props} />;
  }

  const fieldKey = generateHandleId(idSchema.$id ?? "");
  const { nodeId } = formContext;
  const type = mapJsonSchemaTypeToInputType(schema);

  return (
    <InputRenderer
      id={`${name}-input`}
      type={type}
      nodeId={nodeId}
      fieldKey={fieldKey}
      value={formData}
      onChange={onChange}
      placeholder={`Enter ${name || "Contact Data"}`}
    />
  );
};
