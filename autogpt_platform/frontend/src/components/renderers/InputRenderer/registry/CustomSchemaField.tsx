import { getDefaultRegistry } from "@rjsf/core";
import { FieldProps } from "@rjsf/utils";
import { findCustomFieldId } from "../custom/custom-registry";

const { SchemaField } = getDefaultRegistry().fields;

export const CustomSchemaField = (fieldProps: FieldProps) => {
  const { schema, registry } = fieldProps;

  const customFieldId = findCustomFieldId(schema);
  if (customFieldId) {
    const CustomField = registry.fields[customFieldId];
    if (CustomField) {
      return <CustomField {...fieldProps} />;
    }
  }

  return <SchemaField {...fieldProps} />;
};
