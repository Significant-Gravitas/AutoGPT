import { tagTransformer } from "./fix-tags.mjs";

/**
 * Transformer function for orval that prepares the OpenAPI spec for generation:
 * 1. Fixes tags (see fix-tags.mjs)
 * 2. Copies shared responses (`components.responses`) into every operation
 *    that references them, with their inline schemas moved to
 *    `components.schemas` as `<Name>Response`. For a shared response, orval 8
 *    imports the type from `models/<name>` but writes `models/<name>Response`.
 * 3. Keeps one media type per response when a response lists several binary
 *    ones (images, videos). Orval 8 writes one mock per media type under the
 *    same name, which fails to compile; the generated type is `Blob` either way.
 *
 * @param {OpenAPIObject} inputSchema
 * @return {OpenAPIObject}
 */
export const specTransformer = (inputSchema) =>
  collapseBinaryMediaTypes(inlineSharedResponses(tagTransformer(inputSchema)));

function inlineSharedResponses(inputSchema) {
  const { responses = {}, ...components } = inputSchema.components ?? {};
  const schemas = { ...(components.schemas ?? {}) };

  const sharedResponses = Object.fromEntries(
    Object.entries(responses).map(([name, response]) => {
      const content = Object.fromEntries(
        Object.entries(response.content ?? {}).map(([mediaType, media]) => {
          if (!media.schema || media.schema.$ref) return [mediaType, media];
          const schemaName = `${name}Response`;
          if (schemas[schemaName]) {
            throw new Error(
              `fix-spec: schema ${schemaName} already exists; give shared response ${name} a $ref schema instead`,
            );
          }
          schemas[schemaName] = media.schema;
          return [
            mediaType,
            {
              ...media,
              schema: { $ref: `#/components/schemas/${schemaName}` },
            },
          ];
        }),
      );
      return [`#/components/responses/${name}`, { ...response, content }];
    }),
  );

  return {
    ...inputSchema,
    components: { ...components, schemas },
    paths: mapResponses(inputSchema.paths, (response) =>
      response.$ref ? (sharedResponses[response.$ref] ?? response) : response,
    ),
  };
}

function collapseBinaryMediaTypes(inputSchema) {
  return {
    ...inputSchema,
    paths: mapResponses(inputSchema.paths, (response) => {
      const mediaTypes = Object.keys(response.content ?? {});
      if (
        mediaTypes.length < 2 ||
        mediaTypes.some((mediaType) => mediaType.includes("json"))
      ) {
        return response;
      }
      const [first] = mediaTypes;
      return { ...response, content: { [first]: response.content[first] } };
    }),
  };
}

function mapResponses(paths, fn) {
  return Object.fromEntries(
    Object.entries(paths ?? {}).map(([path, pathItem]) => [
      path,
      Object.fromEntries(
        Object.entries(pathItem ?? {}).map(([verb, operation]) => {
          if (!operation?.responses) return [verb, operation];
          const responses = Object.fromEntries(
            Object.entries(operation.responses).map(([status, response]) => [
              status,
              fn(response),
            ]),
          );
          return [verb, { ...operation, responses }];
        }),
      ),
    ]),
  );
}

export default specTransformer;
