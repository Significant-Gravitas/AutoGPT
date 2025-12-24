import { RJSFSchema } from "@rjsf/utils";

export function isAnyOfSchema(schema: RJSFSchema | undefined): boolean {
  return Array.isArray(schema?.anyOf) && schema!.anyOf.length > 0;
}

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
