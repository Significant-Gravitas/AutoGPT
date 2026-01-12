import { getUiOptions, RJSFSchema, UiSchema } from "@rjsf/utils";

export function isAnyOfSchema(schema: RJSFSchema | undefined): boolean {
  return (
    Array.isArray(schema?.anyOf) &&
    schema!.anyOf.length > 0 &&
    schema?.enum === undefined
  );
}

export const isAnyOfChild = (
  uiSchema: UiSchema<any, RJSFSchema, any> | undefined,
): boolean => {
  const uiOptions = getUiOptions(uiSchema);
  return uiOptions.label === false;
};

export function isOptionalType(schema: RJSFSchema | undefined): {
  isOptional: boolean;
  type?: any;
} {
  if (
    !Array.isArray(schema?.anyOf) ||
    schema!.anyOf.length !== 2 ||
    !schema!.anyOf.some((opt: any) => opt.type === "null")
  ) {
    return { isOptional: false };
  }

  const nonNullType = schema!.anyOf?.find((opt: any) => opt.type !== "null");

  return {
    isOptional: true,
    type: nonNullType,
  };
}
export function isAnyOfSelector(name: string) {
  return name.includes("anyof_select");
}

export function isMultiSelectSchema(schema: RJSFSchema | undefined): boolean {
  if (typeof schema !== "object" || schema === null) {
    return false;
  }

  if ("anyOf" in schema || "oneOf" in schema) {
    return false;
  }

  return !!(
    schema.type === "object" &&
    schema.properties &&
    Object.values(schema.properties).every(
      (prop: any) => prop.type === "boolean",
    )
  );
}
