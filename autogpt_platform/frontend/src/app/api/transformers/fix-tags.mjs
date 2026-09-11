/**
 * Transformer function for orval that fixes tags in OpenAPI spec.
 * 1. Create a set of tags so we have unique values
 * 2. Then remove public, private, v1, and v2 tags from tags array
 * 3. Then arrange remaining tags alphabetically and only keep the first one
 *
 * @param {OpenAPIObject} inputSchema
 * @return {OpenAPIObject}
 */

export const tagTransformer = (inputSchema) => {
  const processedPaths = Object.entries(inputSchema.paths || {}).reduce(
    (acc, [path, pathItem]) => ({
      ...acc,
      [path]: Object.entries(pathItem || {}).reduce(
        (pathItemAcc, [verb, operation]) => {
          if (typeof operation === "object" && operation !== null) {
            // 1. Create a set of tags so we have unique values
            const uniqueTags = Array.from(new Set(operation.tags || []));

            // 2. Remove public, private, v1, and v2 tags from tags array
            const filteredTags = uniqueTags.filter(
              (tag) =>
                !["public", "private"].includes(tag.toLowerCase()) &&
                !/^v[12]$/i.test(tag),
            );

            // 3. Arrange tags alphabetically and only keep the first one
            const sortedTags = filteredTags.sort((a, b) => a.localeCompare(b));
            const firstTag = sortedTags.length > 0 ? [sortedTags[0]] : [];

            return {
              ...pathItemAcc,
              [verb]: {
                ...operation,
                tags: firstTag,
              },
            };
          }
          return {
            ...pathItemAcc,
            [verb]: operation,
          };
        },
        {},
      ),
    }),
    {},
  );

  return {
    ...inputSchema,
    paths: processedPaths,
  };
};

/**
 * fastapi>=0.141 renders multipart file fields as
 * `{ type: "string", contentMediaType: "application/octet-stream" }` instead of
 * `{ type: "string", format: "binary" }`. orval maps only `format: "binary"`
 * to `Blob`; the new shape generates `string`, which breaks every upload call
 * site that passes a `File`/`Blob`. Restore `format: "binary"` for orval's
 * view of the spec (the committed openapi.json keeps the backend's output
 * verbatim; this only affects client generation).
 *
 * @param {OpenAPIObject} inputSchema
 * @return {OpenAPIObject}
 */
const binaryUploadTransformer = (inputSchema) => {
  const schemas = inputSchema.components?.schemas;
  if (!schemas) {
    return inputSchema;
  }
  for (const schema of Object.values(schemas)) {
    const props = schema?.properties;
    if (!props) continue;
    for (const prop of Object.values(props)) {
      if (
        prop?.type === "string" &&
        prop?.contentMediaType === "application/octet-stream"
      ) {
        prop.format = "binary";
      }
    }
  }
  return inputSchema;
};

const combinedTransformer = (inputSchema) =>
  binaryUploadTransformer(tagTransformer(inputSchema));

export default combinedTransformer;
