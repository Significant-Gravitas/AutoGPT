import { RJSFSchema } from "@rjsf/utils";

const isArraySchema = (
  schema: RJSFSchema,
): schema is RJSFSchema & { type: "array" } => {
  return schema.type === "array";
};

const isObjectSchema = (
  schema: RJSFSchema,
): schema is RJSFSchema & { type: "object" } => {
  return schema.type === "object";
};

// Helper to detect type and build path segments
export function parseFieldPath(
  rootSchema: RJSFSchema,
  id: string,
  idSeparator: string = "_%_",
): { path: string[]; typeHints: string[] } {
  const segments = id.split(idSeparator).filter(Boolean);
  const typeHints: string[] = [];

  let currentSchema = rootSchema;

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    const isNumeric = /^\d+$/.test(segment);

    if (isNumeric) {
      typeHints.push("array");
    } else {
      // Object property (string key)
      typeHints.push("object-key");
      currentSchema = (currentSchema.properties?.[segment] as RJSFSchema) || {};
    }
  }

  return { path: segments, typeHints };
}

// Helper to generate display path with custom delimiters
export function getHandleId(
  rootSchema: RJSFSchema,
  id: string,
  idSeparator: string = "_%_",
): string {
  const idPrefix = "agpt_";
  const idSuffix = "__title";

  if (id.startsWith(idPrefix)) {
    id = id.slice(idPrefix.length);
  }
  if (id.endsWith(idSuffix)) {
    id = id.slice(0, -idSuffix.length);
  }

  const { path, typeHints } = parseFieldPath(rootSchema, id, idSeparator);

  return path
    .map((seg, i) => {
      const type = typeHints[i];
      const prevType = i > 0 ? typeHints[i - 1] : null;

      if (/^\d+$/.test(seg)) {
        // Numeric segment
        if (type === "array" || prevType === "array") {
          return `$_$${seg}`; // array index
        }
        return `.${seg}`; // object numeric key
      } else {
        return `_${seg}`; // key value pair already contain _#_ delimiter
      }
    })
    .join("");
}
