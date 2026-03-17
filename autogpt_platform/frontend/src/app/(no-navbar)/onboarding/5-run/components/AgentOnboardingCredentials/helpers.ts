import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";

export function getCredentialFields(
  agent: GraphModel | null,
): AgentCredentialsFields {
  if (!agent) return {};

  const hasNoInputs =
    !agent.credentials_input_schema ||
    typeof agent.credentials_input_schema !== "object" ||
    !("properties" in agent.credentials_input_schema) ||
    !agent.credentials_input_schema.properties;

  if (hasNoInputs) return {};

  return agent.credentials_input_schema.properties as AgentCredentialsFields;
}

export type AgentCredentialsFields = Record<
  string,
  BlockIOCredentialsSubSchema
>;

export function areAllCredentialsSet(
  fields: AgentCredentialsFields,
  inputs: Record<string, CredentialsMetaInput | undefined>,
) {
  const required = Object.keys(fields || {});
  return required.every((k) => Boolean(inputs[k]));
}
