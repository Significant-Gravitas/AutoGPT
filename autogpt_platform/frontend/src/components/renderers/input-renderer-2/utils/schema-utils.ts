import { getUiOptions, RJSFSchema, UiSchema } from "@rjsf/utils";

export function isAnyOfSchema(schema: RJSFSchema | undefined): boolean {
  return Array.isArray(schema?.anyOf) && schema!.anyOf.length > 0;
}

export const isAnyOfChild = (
  uiSchema: UiSchema<any, RJSFSchema, any> | undefined,
): boolean => {
  const uiOptions = getUiOptions(uiSchema);
  return uiOptions.label === false;
};

export function isOptionalType(schema: RJSFSchema | undefined): boolean {
  return (
    Array.isArray(schema?.anyOf) &&
    schema!.anyOf.length === 2 &&
    schema!.anyOf.some((opt: any) => opt.type === "null")
  );
}

export function isAnyOfSelector(name: string) {
  return name.includes("anyof_select");
}
