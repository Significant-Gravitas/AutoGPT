import { RJSFSchema } from "@rjsf/utils";

export function parseFieldPath(
  rootSchema: RJSFSchema,
  id: string,
  additional: boolean,
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
      if (additional) {
        typeHints.push("object-key");
      } else {
        typeHints.push("object-property");
      }
      currentSchema = (currentSchema.properties?.[segment] as RJSFSchema) || {};
    }
  }

  return { path: segments, typeHints };
}

// This helper work is simple - it just help us to convert rjsf id to our backend compatible id
// Example : List[dict] = agpt_%_List_0_dict__title -> List_$_0_#_dict
// We remove the prefix and suffix and then we split id by our custom delimiter (_%_)
// then add _$_ delimiter for array and _#_ delimiter for object-key
// and for normal property we add . delimiter

export function getHandleId(
  rootSchema: RJSFSchema,
  id: string,
  additional: boolean,
  idSeparator: string = "_%_",
): string {
  const idPrefix = "agpt_%_";
  const idSuffix = "__title";

  if (id.startsWith(idPrefix)) {
    id = id.slice(idPrefix.length);
  }
  if (id.endsWith(idSuffix)) {
    id = id.slice(0, -idSuffix.length);
  }

  const { path, typeHints } = parseFieldPath(
    rootSchema,
    id,
    additional,
    idSeparator,
  );

  return path
    .map((seg, i) => {
      const type = typeHints[i];
      if (type === "array") {
        return `_$_${seg}`;
      }
      if (type === "object-key") {
        return `_${seg}`; // we haven't added _#_ delimiter for object-key because it's already added in the id - check WrapIfAdditionalTemplate.tsx
      }

      return `.${seg}`;
    })
    .join("")
    .slice(1);
}
