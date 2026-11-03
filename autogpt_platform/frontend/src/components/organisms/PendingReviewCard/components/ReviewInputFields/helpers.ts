import type { BlockIOSubSchema } from "@/lib/autogpt-server-api/types";
import { beautifyString } from "@/lib/utils";

type Schema = Record<string, unknown>;

export type ReviewField =
  | { kind: "credentials"; key: string; label: string; text: string }
  | { kind: "secret"; key: string; label: string }
  | { kind: "group"; key: string; label: string; fields: ReviewField[] }
  | {
      kind: "input";
      key: string;
      label: string;
      schema: BlockIOSubSchema;
      value: unknown;
    }
  | { kind: "raw"; key: string; label: string; value: unknown };

/** The fields to show for a review payload, labelled from the block's input
 *  schema when there is one and from the payload's own keys otherwise. */
export function reviewFields(
  payload: Record<string, unknown>,
  schema?: Schema | null,
): ReviewField[] {
  const properties = (schema?.properties ?? {}) as Record<string, Schema>;
  return Object.entries(payload).map(([key, value]) =>
    toField(key, value, properties[key]),
  );
}

function toField(key: string, value: unknown, schema?: Schema): ReviewField {
  const label = (schema?.title as string | undefined) ?? beautifyString(key);
  if (isCredentials(key, value, schema)) {
    // A credentials field's schema title is a type name, not a label.
    const accountLabel =
      key === "credentials" ? "Account" : beautifyString(key);
    return {
      kind: "credentials",
      key,
      label: accountLabel,
      text: credentialsText(value),
    };
  }
  if (schema?.secret) return { kind: "secret", key, label };
  if (isPlainObject(value)) {
    const properties = (schema?.properties ?? {}) as Record<string, Schema>;
    return {
      kind: "group",
      key,
      label,
      fields: Object.entries(value).map(([k, v]) =>
        toField(k, v, properties[k]),
      ),
    };
  }
  if (Array.isArray(value)) return { kind: "raw", key, label, value };
  return {
    kind: "input",
    key,
    label,
    value,
    schema: {
      ...(schema ?? { type: inferType(value) }),
      title: label,
    } as BlockIOSubSchema,
  };
}

function isCredentials(key: string, value: unknown, schema?: Schema) {
  if (schema && "credentials_provider" in schema) return true;
  return (
    key === "credentials" &&
    isPlainObject(value) &&
    "provider" in value &&
    "id" in value
  );
}

function credentialsText(value: unknown) {
  if (!isPlainObject(value)) return "Connected account";
  const title = typeof value.title === "string" ? value.title : null;
  const provider =
    typeof value.provider === "string" ? beautifyString(value.provider) : null;
  if (title && provider) return `${title} (${provider})`;
  return title ?? provider ?? "Connected account";
}

function inferType(value: unknown) {
  if (typeof value === "number") return "number";
  if (typeof value === "boolean") return "boolean";
  return "string";
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
