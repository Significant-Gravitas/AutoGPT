import {
  RJSFSchema,
  UIOptionsType,
  StrictRJSFSchema,
  FormContextType,
} from "@rjsf/utils";

export const ANY_OF_FLAG = "__anyOf";
export const ARRAY_FLAG = "__array";
export const OBJECT_FLAG = "__object";
export const KEY_PAIR_FLAG = "__keyPair";
export const TITLE_FLAG = "__title";
export const ARRAY_ITEM_FLAG = "__arrayItem";
export const ID_PREFIX = "agpt_%_";

const FLAG_LIST = [OBJECT_FLAG, KEY_PAIR_FLAG, ARRAY_ITEM_FLAG];

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
  if (handleId.includes(ANY_OF_FLAG)) {
    return handleId.replace(ANY_OF_FLAG, "");
  }
  if (handleId.includes(ARRAY_FLAG)) {
    return handleId.replace(ARRAY_FLAG, "_$_");
  }
  if (handleId.includes(OBJECT_FLAG)) {
    return handleId.replace(OBJECT_FLAG, ".");
  }
  if (handleId.includes(KEY_PAIR_FLAG)) {
    return handleId.replace(KEY_PAIR_FLAG, "_#_");
  }
  if (handleId.includes(TITLE_FLAG)) {
    return handleId.replace(TITLE_FLAG, "");
  }
  if (handleId.includes(ID_PREFIX)) {
    return handleId.replace(ID_PREFIX, "");
  }
  return handleId;
};

export const delimitterForFlag = (flag: string) => {
  if (flag === ARRAY_FLAG) {
    return "_$_";
  }
  if (flag === OBJECT_FLAG) {
    return ".";
  }
  if (flag === KEY_PAIR_FLAG) {
    return "_#_";
  }
  return "";
};

export const getHandleId = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(
  uiOptions: UIOptionsType<T, S, F>,
  id: string,
) => {
  const parentHandleId = uiOptions.handleId as string;

  if (!parentHandleId) {
    return cleanUpHandleId(id);
  }

  const prefixToAdd = id.split("_%_").pop();
  const flag = parentHandleId.split("_%_").pop();

  if (!flag) {
    return cleanUpHandleId(id);
  }

  if (FLAG_LIST.includes(flag)) {
    return parentHandleId + delimitterForFlag(flag) + prefixToAdd;
  }

  return parentHandleId;
};
