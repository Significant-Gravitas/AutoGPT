import { z } from "zod/v4";
import { catalog } from "./catalog";

export function validateWorkspace(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(validateWorkspace);
  if (!value || typeof value !== "object") return value;
  if (
    "typeName" in value &&
    typeof value.typeName === "string" &&
    "props" in value
  ) {
    const component = catalog.components[value.typeName];
    if (!component) throw new Error("Unsupported workspace component.");
    const props = validateWorkspace(value.props);
    z.parse(component.props, props);
    return props;
  }
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [key, validateWorkspace(item)]),
  );
}
