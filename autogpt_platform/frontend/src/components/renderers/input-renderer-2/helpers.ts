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
} from "./constants";

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

export const isKeyValuePair = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
  schema,
}: {
  schema: RJSFSchema;
}) => {
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
    return cleanUpHandleId(id);
  }

  if (isPartOfAnyOf({ uiOptions })) {
    return cleanUpHandleId(parentHandleId);
  }

  if (isKeyValuePair({ schema })) {
    const key = id.split("_%_").at(-1);
    const prefix = id.split("_%_").slice(0, -1).join("_%_");

    const handleId = `${prefix}_#_${key}`;
    return cleanUpHandleId(handleId);
  }

  if (isArrayItem({ uiOptions })) {
    const index = id.split("_%_").at(-1);
    const prefix = id.split("_%_").slice(0, -1).join("_%_");
    const handleId = `${prefix}_$_${index}`;
    return cleanUpHandleId(handleId);
  }

  if (isObjectProperty({ uiOptions, schema })) {
    const key = id.split("_%_").at(-1);
    const prefix = id.split("_%_").slice(0, -1).join("_%_");
    const handleId = `${prefix}_@_${key}`;
    return cleanUpHandleId(handleId);
  }
};

export function isCredentialFieldSchema(schema: any): boolean {
  return (
    typeof schema === "object" &&
    schema !== null &&
    "credentials_provider" in schema
  );
}
