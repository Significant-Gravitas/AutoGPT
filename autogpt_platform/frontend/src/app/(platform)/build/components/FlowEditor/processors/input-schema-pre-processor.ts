import { RJSFSchema } from "@rjsf/utils";

/**
 * Pre-processes the input schema to ensure all properties have a type defined.
 * If a property doesn't have a type, it assigns a union of all supported JSON Schema types.
 */
export function preprocessInputSchema(schema: RJSFSchema): RJSFSchema {
  if (!schema || typeof schema !== "object") {
    return schema;
  }

  const processedSchema = { ...schema };

  // Recursively process properties
  if (processedSchema.properties) {
    processedSchema.properties = { ...processedSchema.properties };

    for (const [key, property] of Object.entries(processedSchema.properties)) {
      if (property && typeof property === "object") {
        const processedProperty = { ...property };

        // Only add type if no type is defined AND no anyOf/oneOf/allOf is present
        if (
          !processedProperty.type &&
          !processedProperty.anyOf &&
          !processedProperty.oneOf &&
          !processedProperty.allOf
        ) {
          processedProperty.anyOf = [
            { type: "string" },
            { type: "number" },
            { type: "integer" },
            { type: "boolean" },
            { type: "array", items: { type: "string" } },
            { type: "object" },
            { type: "null" },
          ];
        }

        // when encountering an array with items missing type
        if (processedProperty.type === "array" && processedProperty.items) {
          const items = processedProperty.items as RJSFSchema;
          if (!items.type && !items.anyOf && !items.oneOf && !items.allOf) {
            processedProperty.items = {
              type: "string",
              title: items.title ?? "",
            };
          } else {
            processedProperty.items = preprocessInputSchema(items);
          }
        }

        // Recursively process nested objects
        if (
          processedProperty.type === "object" ||
          (Array.isArray(processedProperty.type) &&
            processedProperty.type.includes("object"))
        ) {
          processedProperty.properties = processProperties(
            processedProperty.properties,
          );
        }

        // Process array items
        if (
          processedProperty.type === "array" ||
          (Array.isArray(processedProperty.type) &&
            processedProperty.type.includes("array"))
        ) {
          if (processedProperty.items) {
            processedProperty.items = preprocessInputSchema(
              processedProperty.items as RJSFSchema,
            );
          }
        }

        processedSchema.properties[key] = processedProperty;
      }
    }
  }

  // Process array items at root level
  if (processedSchema.items) {
    processedSchema.items = preprocessInputSchema(
      processedSchema.items as RJSFSchema,
    );
  }

  processedSchema.title = ""; // Otherwise our form creator will show the title of the schema in the input field
  processedSchema.description = ""; // Otherwise our form creator will show the description of the schema in the input field

  return processedSchema;
}

/**
 * Helper function to process properties object
 */
function processProperties(properties: any): any {
  if (!properties || typeof properties !== "object") {
    return properties;
  }

  const processedProperties = { ...properties };

  for (const [key, property] of Object.entries(processedProperties)) {
    if (property && typeof property === "object") {
      processedProperties[key] = preprocessInputSchema(property as RJSFSchema);
    }
  }

  return processedProperties;
}
