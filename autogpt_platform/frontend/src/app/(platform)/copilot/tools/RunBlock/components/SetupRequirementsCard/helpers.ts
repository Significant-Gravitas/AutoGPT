import type { CredentialField } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";
import type { RJSFSchema } from "@rjsf/utils";

const VALID_CREDENTIAL_TYPES = new Set([
  "api_key",
  "oauth2",
  "user_password",
  "host_scoped",
]);

export function coerceCredentialFields(rawMissingCredentials: unknown): {
  credentialFields: CredentialField[];
  requiredCredentials: Set<string>;
} {
  const missing =
    rawMissingCredentials && typeof rawMissingCredentials === "object"
      ? (rawMissingCredentials as Record<string, unknown>)
      : {};

  const credentialFields: CredentialField[] = [];
  const requiredCredentials = new Set<string>();

  Object.entries(missing).forEach(([key, value]) => {
    if (!value || typeof value !== "object") return;
    const cred = value as Record<string, unknown>;

    const provider =
      typeof cred.provider === "string" ? cred.provider.trim() : "";
    if (!provider) return;

    const types =
      Array.isArray(cred.types) && cred.types.length > 0 ? cred.types : [];

    const credentialTypes = types
      .map((t) => (typeof t === "string" ? t.trim() : ""))
      .filter((t) => VALID_CREDENTIAL_TYPES.has(t));

    if (credentialTypes.length === 0) return;

    const scopes = Array.isArray(cred.scopes)
      ? cred.scopes.filter((s): s is string => typeof s === "string")
      : undefined;

    const discriminator =
      typeof cred.discriminator === "string" ? cred.discriminator : undefined;
    const discriminatorValues = Array.isArray(cred.discriminator_values)
      ? cred.discriminator_values.filter(
          (v): v is string => typeof v === "string",
        )
      : undefined;

    const schema: Record<string, unknown> = {
      type: "object" as const,
      properties: {},
      credentials_provider: [provider],
      credentials_types: credentialTypes,
      credentials_scopes: scopes,
    };

    if (discriminator) {
      schema.discriminator = discriminator;
    }
    if (discriminatorValues && discriminatorValues.length > 0) {
      schema.discriminator_values = discriminatorValues;
    }

    credentialFields.push([key, schema]);
    requiredCredentials.add(key);
  });

  return { credentialFields, requiredCredentials };
}

/**
 * Build a sibling-inputs dict from the missing_credentials discriminator values.
 *
 * When the backend resolves credentials for host-scoped blocks (e.g.
 * SendAuthenticatedWebRequestBlock), it adds the target URL to
 * `discriminator_values`.  The credential modal uses `siblingInputs`
 * to extract the host and prefill the "Host Pattern" field.
 *
 * This function builds that mapping from the `discriminator` field name
 * and the first `discriminator_values` entry for each credential.
 */
export function buildSiblingInputsFromCredentials(
  rawMissingCredentials: unknown,
): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  if (!rawMissingCredentials || typeof rawMissingCredentials !== "object")
    return result;

  const missing = rawMissingCredentials as Record<string, unknown>;
  for (const value of Object.values(missing)) {
    if (!value || typeof value !== "object") continue;
    const cred = value as Record<string, unknown>;

    const discriminator =
      typeof cred.discriminator === "string" ? cred.discriminator : null;
    const discriminatorValues = Array.isArray(cred.discriminator_values)
      ? cred.discriminator_values.filter(
          (v): v is string => typeof v === "string",
        )
      : [];

    if (discriminator && discriminatorValues.length > 0) {
      result[discriminator] = discriminatorValues[0];
    }
  }

  return result;
}

interface ExpectedInput {
  name: string;
  title: string;
  type: string;
  description?: string;
  required: boolean;
  advanced: boolean;
  value?: unknown;
}

export function coerceExpectedInputs(rawInputs: unknown): ExpectedInput[] {
  if (!Array.isArray(rawInputs)) return [];
  const results: ExpectedInput[] = [];

  rawInputs.forEach((value, index) => {
    if (!value || typeof value !== "object") return;
    const input = value as Record<string, unknown>;

    const name =
      typeof input.name === "string" && input.name.trim()
        ? input.name.trim()
        : `input-${index}`;
    const title =
      typeof input.title === "string" && input.title.trim()
        ? input.title.trim()
        : name;
    const type = typeof input.type === "string" ? input.type : "unknown";
    const description =
      typeof input.description === "string" && input.description.trim()
        ? input.description.trim()
        : undefined;
    const required = Boolean(input.required);
    const advanced = Boolean(input.advanced);

    const item: ExpectedInput = { name, title, type, required, advanced };
    if (description) item.description = description;
    if (input.value !== undefined && input.value !== null) {
      item.value = input.value;
    }
    results.push(item);
  });

  return results;
}

/**
 * Build an RJSF schema from expected inputs so they can be rendered
 * as a dynamic form via FormRenderer.
 *
 * When ``showAdvanced`` is false (default), fields marked ``advanced``
 * are excluded — matching the builder behaviour where advanced fields
 * are hidden behind a toggle.
 */
export function buildExpectedInputsSchema(
  expectedInputs: ExpectedInput[],
  showAdvanced = false,
): RJSFSchema | null {
  const visible = showAdvanced
    ? expectedInputs
    : expectedInputs.filter((i) => !i.advanced);

  if (visible.length === 0) return null;

  const TYPE_MAP: Record<string, string> = {
    string: "string",
    str: "string",
    text: "string",
    number: "number",
    int: "integer",
    integer: "integer",
    float: "number",
    boolean: "boolean",
    bool: "boolean",
  };

  const properties: Record<string, Record<string, unknown>> = {};
  const required: string[] = [];

  for (const input of visible) {
    const prop: Record<string, unknown> = {
      type: TYPE_MAP[input.type.toLowerCase()] ?? "string",
      title: input.title,
    };
    if (input.description) prop.description = input.description;
    if (input.value !== undefined) prop.default = input.value;
    properties[input.name] = prop;
    if (input.required) required.push(input.name);
  }

  return {
    type: "object",
    properties,
    ...(required.length > 0 ? { required } : {}),
  };
}

/**
 * Extract initial form values from expected inputs that have a
 * prefilled ``value`` from the backend.
 */
export function extractInitialValues(
  expectedInputs: ExpectedInput[],
): Record<string, unknown> {
  const values: Record<string, unknown> = {};
  for (const input of expectedInputs) {
    if (input.value !== undefined && input.value !== null) {
      values[input.name] = input.value;
    }
  }
  return values;
}

export function mergeInputValues(
  initialValues: Record<string, unknown>,
  prev: Record<string, unknown>,
): Record<string, unknown> {
  const merged = { ...initialValues };
  for (const [key, value] of Object.entries(prev)) {
    if (value !== undefined && value !== null && value !== "") {
      merged[key] = value;
    }
  }
  return merged;
}

export function checkAllCredentialsComplete(
  requiredCredentials: Set<string>,
  inputCredentials: Record<string, unknown>,
): boolean {
  return [...requiredCredentials].every((key) => !!inputCredentials[key]);
}

export function getRequiredInputNames(
  expectedInputs: ExpectedInput[],
): string[] {
  return expectedInputs
    .filter((i) => i.required && !i.advanced)
    .map((i) => i.name);
}

export function checkAllInputsComplete(
  expectedInputs: ExpectedInput[],
  inputValues: Record<string, unknown>,
): boolean {
  if (expectedInputs.length === 0) return true;
  const requiredNames = getRequiredInputNames(expectedInputs);
  return requiredNames.every((name) => {
    const v = inputValues[name];
    return v !== undefined && v !== null && v !== "";
  });
}

export function checkCanRun(
  needsCredentials: boolean,
  isAllCredentialsComplete: boolean,
  isAllInputsComplete: boolean,
): boolean {
  return (!needsCredentials || isAllCredentialsComplete) && isAllInputsComplete;
}

export function buildRunMessage(
  needsCredentials: boolean,
  needsInputs: boolean,
  inputValues: Record<string, unknown>,
  retryInstruction?: string,
): string {
  const parts: string[] = [];
  if (needsCredentials) {
    parts.push("I've configured the required credentials.");
  }

  if (needsInputs) {
    const nonEmpty = Object.fromEntries(
      Object.entries(inputValues).filter(
        ([, v]) => v !== undefined && v !== null && v !== "",
      ),
    );
    parts.push(`Run with these inputs: ${JSON.stringify(nonEmpty, null, 2)}`);
  } else {
    parts.push(retryInstruction ?? "Please re-run this step now.");
  }

  return parts.join(" ");
}
