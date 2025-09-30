import React from "react";
import { FieldProps } from "@rjsf/utils";
import { getDefaultRegistry } from "@rjsf/core";
import { generateHandleId } from "../../handlers/helpers";
import { ObjectEditor } from "../../components/ObjectEditor/ObjectEditor";

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

  return (
    <ObjectEditor
      id={`${name}-input`}
      nodeId={nodeId}
      fieldKey={fieldKey}
      value={formData}
      onChange={onChange}
      placeholder={`Enter ${name || "Contact Data"}`}
    />
  );
};
