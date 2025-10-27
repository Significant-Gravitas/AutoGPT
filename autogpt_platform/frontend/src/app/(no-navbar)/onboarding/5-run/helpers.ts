import type { GraphMeta } from "@/lib/autogpt-server-api";
import type {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import type { InputValues } from "./types";

export function computeInitialAgentInputs(
  agent: GraphMeta | null,
  existingInputs?: InputValues | null,
): InputValues {
  const properties = agent?.input_schema?.properties || {};
  const result: InputValues = {};

  Object.entries(properties).forEach(([key, subSchema]) => {
    if (
      existingInputs &&
      key in existingInputs &&
      existingInputs[key] != null
    ) {
      result[key] = existingInputs[key];
      return;
    }
    // GraphIOSubSchema.default is typed as string, but server may return other primitives
    const def = (subSchema as unknown as { default?: string | number }).default;
    result[key] = def ?? "";
  });

  return result;
}

export function getAgentCredentialsInputFields(agent: GraphMeta | null) {
  const hasNoInputs =
    !agent?.credentials_input_schema ||
    typeof agent.credentials_input_schema !== "object" ||
    !("properties" in agent.credentials_input_schema) ||
    !agent.credentials_input_schema.properties;

  if (hasNoInputs) return {};

  return agent.credentials_input_schema.properties;
}

export function areAllCredentialsSet(
  fields: Record<string, BlockIOCredentialsSubSchema>,
  inputs: Record<string, CredentialsMetaInput | undefined>,
) {
  const required = Object.keys(fields || {});
  return required.every((k) => Boolean(inputs[k]));
}

type IsRunDisabledParams = {
  agent: GraphMeta | null;
  isRunning: boolean;
  agentInputs: InputValues | null | undefined;
  credentialsRequired: boolean;
  credentialsSatisfied: boolean;
};

export function isRunDisabled({
  agent,
  isRunning,
  agentInputs,
  credentialsRequired,
  credentialsSatisfied,
}: IsRunDisabledParams) {
  const hasEmptyInput = Object.values(agentInputs || {}).some(
    (value) => String(value).trim() === "",
  );

  if (hasEmptyInput) return true;
  if (!agent) return true;
  if (isRunning) return true;
  if (credentialsRequired && !credentialsSatisfied) return true;

  return false;
}

export function getSchemaDefaultCredentials(
  schema: BlockIOCredentialsSubSchema,
): CredentialsMetaInput | undefined {
  return schema.default as CredentialsMetaInput | undefined;
}

export function sanitizeCredentials(
  map: Record<string, CredentialsMetaInput | undefined>,
): Record<string, CredentialsMetaInput> {
  const sanitized: Record<string, CredentialsMetaInput> = {};
  for (const [key, value] of Object.entries(map)) {
    if (value) sanitized[key] = value;
  }
  return sanitized;
}
