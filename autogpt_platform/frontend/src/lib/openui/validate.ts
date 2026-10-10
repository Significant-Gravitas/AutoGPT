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
    if (!component)
      throw new Error(`Unsupported workspace component: ${value.typeName}.`);
    const props = validateWorkspace(value.props);
    const parsed = z.safeParse(component.props, props);
    if (!parsed.success) {
      const issues = parsed.error.issues
        .slice(0, 3)
        .map(
          (issue) =>
            `${value.typeName}.${issue.path.join(".")}: ${issue.message}`,
        );
      throw new Error(issues.join("; "));
    }
    if (value.typeName === "Form") validateFieldNames(props);
    return props;
  }
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [key, validateWorkspace(item)]),
  );
}

function validateFieldNames(props: unknown) {
  const { fields } = z.parse(
    z.object({ fields: z.array(z.object({ name: z.string() })) }),
    props,
  );
  if (new Set(fields.map((field) => field.name)).size !== fields.length) {
    throw new Error("Form.fields: Field names must be unique within a form.");
  }
}
