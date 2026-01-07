import { RJSFSchema, UiSchema } from "@rjsf/utils";
import { findCustomFieldId } from "../custom/custom-registry";

/**
 * Generates uiSchema with ui:field settings for custom fields based on schema matchers.
 * This is the standard RJSF way to route fields to custom components.
 */
export function generateUiSchemaForCustomFields(
  schema: RJSFSchema,
  existingUiSchema: UiSchema = {},
): UiSchema {
  const uiSchema: UiSchema = { ...existingUiSchema };

  if (schema.properties) {
    for (const [key, propSchema] of Object.entries(schema.properties)) {
      if (propSchema && typeof propSchema === "object") {
        const customFieldId = findCustomFieldId(propSchema);

        if (customFieldId) {
          // Set ui:field to route to the custom field component
          uiSchema[key] = {
            ...(uiSchema[key] as object),
            "ui:field": customFieldId,
          };
        }

        // Recursively handle nested objects
        if (
          propSchema.type === "object" &&
          propSchema.properties &&
          typeof propSchema.properties === "object"
        ) {
          const nestedUiSchema = generateUiSchemaForCustomFields(
            propSchema as RJSFSchema,
            (uiSchema[key] as UiSchema) || {},
          );
          uiSchema[key] = {
            ...(uiSchema[key] as object),
            ...nestedUiSchema,
          };
        }

        // Handle array items
        if (propSchema.type === "array" && propSchema.items) {
          const itemsSchema = propSchema.items as RJSFSchema;
          if (itemsSchema && typeof itemsSchema === "object") {
            const itemsCustomFieldId = findCustomFieldId(itemsSchema);
            if (itemsCustomFieldId) {
              uiSchema[key] = {
                ...(uiSchema[key] as object),
                items: {
                  "ui:field": itemsCustomFieldId,
                },
              };
            } else if (itemsSchema.properties) {
              // Recursively handle array item properties
              const itemsUiSchema = generateUiSchemaForCustomFields(
                itemsSchema,
                ((uiSchema[key] as UiSchema)?.items as UiSchema) || {},
              );
              if (Object.keys(itemsUiSchema).length > 0) {
                uiSchema[key] = {
                  ...(uiSchema[key] as object),
                  items: itemsUiSchema,
                };
              }
            }
          }
        }
      }
    }
  }

  return uiSchema;
}

