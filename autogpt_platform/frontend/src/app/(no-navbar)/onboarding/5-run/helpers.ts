import type {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import type { InputValues } from "./types";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";

export function computeInitialAgentInputs(
  agent: GraphModel | null,
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
    const def = (subSchema as unknown as { default?: string | number }).default;
    result[key] = def ?? "";
  });

  return result;
}

type IsRunDisabledParams = {
  agent: GraphModel | null;
  isRunning: boolean;
  agentInputs: InputValues | null | undefined;
};

export function isRunDisabled({
  agent,
  isRunning,
  agentInputs,
}: IsRunDisabledParams) {
  const hasEmptyInput = Object.values(agentInputs || {}).some(
    (value) => String(value).trim() === "",
  );

  if (hasEmptyInput) return true;
  if (!agent) return true;
  if (isRunning) return true;

  return false;
}

export function getSchemaDefaultCredentials(
  schema: BlockIOCredentialsSubSchema,
): CredentialsMetaInput | undefined {
  return schema.default as CredentialsMetaInput | undefined;
}
