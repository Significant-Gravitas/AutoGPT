import { RJSFSchema, UiSchema } from "@rjsf/utils";
import {
  findCustomFieldId,
  JSON_TEXT_FIELD_ID,
} from "../custom/custom-registry";

function isComplexType(schema: RJSFSchema): boolean {
  return schema.type === "object" || schema.type === "array";
}

function hasComplexAnyOfOptions(schema: RJSFSchema): boolean {
  const options = schema.anyOf || schema.oneOf;
  if (!Array.isArray(options)) return false;
  return options.some(
    (opt: any) =>
      opt &&
      typeof opt === "object" &&
      (opt.type === "object" || opt.type === "array"),
  );
}

/**
 * Generates uiSchema with ui:field settings for custom fields based on schema matchers.
 * This is the standard RJSF way to route fields to custom components.
 *
 * Nested complex types (arrays/objects inside arrays/objects) are rendered as JsonTextField
 * to avoid deeply nested form UIs. Users can enter raw JSON for these fields.
 *
 * @param schema - The JSON schema
 * @param existingUiSchema - Existing uiSchema to merge with
 * @param insideComplexType - Whether we're already inside a complex type (object/array)
 */
export function generateUiSchemaForCustomFields(
  schema: RJSFSchema,
  existingUiSchema: UiSchema = {},
  insideComplexType: boolean = false,
): UiSchema {
  const uiSchema: UiSchema = { ...existingUiSchema };

  if (schema.properties) {
    for (const [key, propSchema] of Object.entries(schema.properties)) {
      if (propSchema && typeof propSchema === "object") {
        // First check for custom field matchers (credentials, google drive, etc.)
        const customFieldId = findCustomFieldId(propSchema);

        if (customFieldId) {
          uiSchema[key] = {
            ...(uiSchema[key] as object),
            "ui:field": customFieldId,
          };
          // Skip further processing for custom fields
          continue;
        }

        // Handle nested complex types - render as JsonTextField
        if (insideComplexType && isComplexType(propSchema as RJSFSchema)) {
          uiSchema[key] = {
            ...(uiSchema[key] as object),
            "ui:field": JSON_TEXT_FIELD_ID,
          };
          // Don't recurse further - this field is now a text input
          continue;
        }

        // Handle anyOf/oneOf inside complex types
        if (
          insideComplexType &&
          hasComplexAnyOfOptions(propSchema as RJSFSchema)
        ) {
          uiSchema[key] = {
            ...(uiSchema[key] as object),
            "ui:field": JSON_TEXT_FIELD_ID,
          };
          continue;
        }

        // Recurse into object properties
        if (
          propSchema.type === "object" &&
          propSchema.properties &&
          typeof propSchema.properties === "object"
        ) {
          const nestedUiSchema = generateUiSchemaForCustomFields(
            propSchema as RJSFSchema,
            (uiSchema[key] as UiSchema) || {},
            true, // Now inside a complex type
          );
          uiSchema[key] = {
            ...(uiSchema[key] as object),
            ...nestedUiSchema,
          };
        }

        // Handle arrays
        if (propSchema.type === "array" && propSchema.items) {
          const itemsSchema = propSchema.items as RJSFSchema;
          if (itemsSchema && typeof itemsSchema === "object") {
            // Check for custom field on array items
            const itemsCustomFieldId = findCustomFieldId(itemsSchema);
            if (itemsCustomFieldId) {
              uiSchema[key] = {
                ...(uiSchema[key] as object),
                items: {
                  "ui:field": itemsCustomFieldId,
                },
              };
            } else if (isComplexType(itemsSchema)) {
              // Array items that are complex types become JsonTextField
              uiSchema[key] = {
                ...(uiSchema[key] as object),
                items: {
                  "ui:field": JSON_TEXT_FIELD_ID,
                },
              };
            } else if (hasComplexAnyOfOptions(itemsSchema)) {
              // Array items with anyOf containing complex types become JsonTextField
              uiSchema[key] = {
                ...(uiSchema[key] as object),
                items: {
                  "ui:field": JSON_TEXT_FIELD_ID,
                },
              };
            } else if (itemsSchema.properties) {
              // Recurse into object items (but they're now inside a complex type)
              const itemsUiSchema = generateUiSchemaForCustomFields(
                itemsSchema,
                ((uiSchema[key] as UiSchema)?.items as UiSchema) || {},
                true, // Inside complex type (array)
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

        // Handle anyOf/oneOf at root level - process complex options
        if (!insideComplexType) {
          const anyOfOptions = propSchema.anyOf || propSchema.oneOf;

          if (Array.isArray(anyOfOptions)) {
            for (let i = 0; i < anyOfOptions.length; i++) {
              const option = anyOfOptions[i] as RJSFSchema;
              if (option && typeof option === "object") {
                // Handle anyOf array options with complex items
                if (option.type === "array" && option.items) {
                  const itemsSchema = option.items as RJSFSchema;
                  if (itemsSchema && typeof itemsSchema === "object") {
                    // Array items that are complex types become JsonTextField
                    if (isComplexType(itemsSchema)) {
                      uiSchema[key] = {
                        ...(uiSchema[key] as object),
                        items: {
                          "ui:field": JSON_TEXT_FIELD_ID,
                        },
                      };
                    } else if (hasComplexAnyOfOptions(itemsSchema)) {
                      uiSchema[key] = {
                        ...(uiSchema[key] as object),
                        items: {
                          "ui:field": JSON_TEXT_FIELD_ID,
                        },
                      };
                    }
                  }
                }

                // Recurse into anyOf object options with properties
                if (
                  option.type === "object" &&
                  option.properties &&
                  typeof option.properties === "object"
                ) {
                  const optionUiSchema = generateUiSchemaForCustomFields(
                    option,
                    {},
                    true, // Inside complex type (anyOf object option)
                  );
                  if (Object.keys(optionUiSchema).length > 0) {
                    // Store under the property key - RJSF will apply it
                    uiSchema[key] = {
                      ...(uiSchema[key] as object),
                      ...optionUiSchema,
                    };
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return uiSchema;
}
