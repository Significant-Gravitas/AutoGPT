import React from "react";
import { FieldProps } from "@rjsf/utils";
import { getDefaultRegistry } from "@rjsf/core";
import { generateHandleId } from "@/app/(platform)/build/components/FlowEditor/handlers/helpers";
import { ObjectEditor } from "../widgets/ObjectEditorWidget/ObjectEditorWidget";

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
  let isFreeForm = false;
  if ("additionalProperties" in schema || !("properties" in schema)) {
    isFreeForm = true;
  }

  if (idSchema?.$id === "root" || !isFreeForm) {
    // TODO : We need to create better one
    return <DefaultObjectField {...props} />;
  }

  const fieldKey = generateHandleId(idSchema.$id ?? "");
  const { nodeId } = formContext;

  return (
    <ObjectEditor
      id={idSchema?.$id ?? ""}
      nodeId={nodeId}
      fieldKey={fieldKey}
      value={formData}
      onChange={onChange}
      placeholder={`Enter ${name || "Contact Data"}`}
    />
  );
};
