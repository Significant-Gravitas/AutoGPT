import {
  RJSFSchema,
  UIOptionsType,
  StrictRJSFSchema,
  FormContextType,
  ADDITIONAL_PROPERTY_FLAG,
} from "@rjsf/utils";

import {
  ANY_OF_FLAG,
  ARRAY_ITEM_FLAG,
  ID_PREFIX,
  ID_PREFIX_ARRAY,
  KEY_PAIR_FLAG,
  OBJECT_FLAG,
} from "./constants";
import { PathSegment } from "./types";

export function updateUiOption<T extends Record<string, any>>(
  uiSchema: T | undefined,
  options: Record<string, any>,
): T & { "ui:options": Record<string, any> } {
  return {
    ...(uiSchema || {}),
    "ui:options": {
      ...uiSchema?.["ui:options"],
      ...options,
    },
  } as T & { "ui:options": Record<string, any> };
}

export const cleanUpHandleId = (handleId: string) => {
  let newHandleId = handleId;
  if (handleId.includes(ANY_OF_FLAG)) {
    newHandleId = newHandleId.replace(ANY_OF_FLAG, "");
  }
  if (handleId.includes(ARRAY_ITEM_FLAG)) {
    newHandleId = newHandleId.replace(ARRAY_ITEM_FLAG, "");
  }
  if (handleId.includes(KEY_PAIR_FLAG)) {
    newHandleId = newHandleId.replace(KEY_PAIR_FLAG, "");
  }
  if (handleId.includes(OBJECT_FLAG)) {
    newHandleId = newHandleId.replace(OBJECT_FLAG, "");
  }
  if (handleId.includes(ID_PREFIX_ARRAY)) {
    newHandleId = newHandleId.replace(ID_PREFIX_ARRAY, "");
  }
  if (handleId.includes(ID_PREFIX)) {
    newHandleId = newHandleId.replace(ID_PREFIX, "");
  }
  return newHandleId;
};

export const isArrayItem = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
  uiOptions,
}: {
  uiOptions: UIOptionsType<T, S, F>;
}) => {
  return uiOptions.handleId?.endsWith(ARRAY_ITEM_FLAG);
};

export const isKeyValuePair = ({ schema }: { schema: RJSFSchema }) => {
  return ADDITIONAL_PROPERTY_FLAG in schema;
};

export const isNormal = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
  uiOptions,
}: {
  uiOptions: UIOptionsType<T, S, F>;
}) => {
  return uiOptions.handleId === undefined;
};

export const isPartOfAnyOf = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
  uiOptions,
}: {
  uiOptions: UIOptionsType<T, S, F>;
}) => {
  return uiOptions.handleId?.endsWith(ANY_OF_FLAG);
};
export const isObjectProperty = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
  uiOptions,
  schema,
}: {
  uiOptions: UIOptionsType<T, S, F>;
  schema: RJSFSchema;
}) => {
  return (
    !isArrayItem({ uiOptions }) &&
    !isKeyValuePair({ schema }) &&
    !isNormal({ uiOptions }) &&
    !isPartOfAnyOf({ uiOptions })
  );
};

export const getHandleId = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
  id,
  schema,
  uiOptions,
}: {
  id: string;
  schema: RJSFSchema;
  uiOptions: UIOptionsType<T, S, F>;
}) => {
  const parentHandleId = uiOptions.handleId;

  if (isNormal({ uiOptions })) {
    return id;
  }

  if (isPartOfAnyOf({ uiOptions })) {
    return parentHandleId + ANY_OF_FLAG;
  }

  if (isKeyValuePair({ schema })) {
    const key = id.split("_%_").at(-1);
    let prefix = "";
    if (parentHandleId) {
      prefix = parentHandleId;
    } else {
      prefix = id.split("_%_").slice(0, -1).join("_%_");
    }

    const handleId = `${prefix}_#_${key}`;
    return handleId + KEY_PAIR_FLAG;
  }

  if (isArrayItem({ uiOptions })) {
    const index = id.split("_%_").at(-1);
    const prefix = id.split("_%_").slice(0, -1).join("_%_");
    const handleId = `${prefix}_$_${index}`;
    return handleId + ARRAY_ITEM_FLAG;
  }

  if (isObjectProperty({ uiOptions, schema })) {
    const key = id.split("_%_").at(-1);
    const prefix = id.split("_%_").slice(0, -1).join("_%_");
    const handleId = `${prefix}_@_${key}`;
    return handleId + OBJECT_FLAG;
  }
  return parentHandleId;
};

export function isCredentialFieldSchema(schema: any): boolean {
  return (
    typeof schema === "object" &&
    schema !== null &&
    "credentials_provider" in schema
  );
}

export function parseHandleIdToPath(handleId: string): PathSegment[] {
  const cleanedId = cleanUpHandleId(handleId);
  const segments: PathSegment[] = [];
  const parts = cleanedId.split(/(_#_|_@_|_\$_|\.)/);

  let currentType: "property" | "item" | "additional" | "normal" = "normal";

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];

    if (part === "_#_") {
      currentType = "additional";
    } else if (part === "_@_") {
      currentType = "property";
    } else if (part === "_$_") {
      currentType = "item";
    } else if (part === ".") {
      currentType = "normal";
    } else if (part) {
      const isNumeric = /^\d+$/.test(part);
      if (currentType === "item" && isNumeric) {
        segments.push({
          key: part,
          type: "item",
          index: parseInt(part, 10),
        });
      } else {
        segments.push({
          key: part,
          type: currentType,
        });
      }
      currentType = "normal";
    }
  }

  return segments;
}

/**
 * Ensure a path exists in an object, creating intermediate objects/arrays as needed
 * Returns true if any modifications were made
 */
export function ensurePathExists(
  obj: Record<string, any>,
  segments: PathSegment[],
): boolean {
  if (segments.length === 0) return false;

  let current = obj;
  let modified = false;

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    const isLast = i === segments.length - 1;
    const nextSegment = segments[i + 1];

    const getDefaultValue = () => {
      if (isLast) {
        return "";
      }
      if (nextSegment?.type === "item") {
        return [];
      }
      return {};
    };

    if (segment.type === "item" && segment.index !== undefined) {
      if (!Array.isArray(current)) {
        return modified;
      }

      while (current.length <= segment.index) {
        current.push(isLast ? "" : {});
        modified = true;
      }

      if (!isLast) {
        if (
          current[segment.index] === undefined ||
          current[segment.index] === null
        ) {
          current[segment.index] = getDefaultValue();
          modified = true;
        }
        current = current[segment.index];
      }
    } else {
      if (!(segment.key in current)) {
        current[segment.key] = getDefaultValue();
        modified = true;
      } else if (!isLast && current[segment.key] === undefined) {
        current[segment.key] = getDefaultValue();
        modified = true;
      }

      if (!isLast) {
        current = current[segment.key];
      }
    }
  }

  return modified;
}
