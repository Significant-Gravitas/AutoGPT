import type { RJSFSchema } from "@rjsf/utils";
import type {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api";
import type { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";
import {
  classifyCredentials,
  getSupportedCredentialTypes,
} from "@/hooks/useCredentials";
import {
  getCredentialProviderFromSchema,
  getDiscriminatorValue,
  isCredentialFieldSchema,
} from "@/components/renderers/InputRenderer/custom/CredentialField/helpers";
import { getUsableKeys } from "@/components/contextual/CredentialsInput/helpers";
import { toCredentialsInput } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/node-credentials";

/** Resolve defaults in the edit that adds a step or changes its model/transport. */
export function resolveStepCredentials(
  schema: RJSFSchema,
  inputs: Record<string, unknown>,
  providers: CredentialsProvidersContextType | null,
  previousInputs?: Record<string, unknown>,
) {
  const result = { ...inputs };
  for (const [name, property] of Object.entries(schema.properties ?? {})) {
    if (!isCredentialFieldSchema(property)) continue;
    const field = property as BlockIOCredentialsSubSchema;
    const discriminator = field.discriminator;
    const routeSchema = discriminator && schema.properties?.[discriminator];
    const defaults =
      discriminator && typeof routeSchema === "object"
        ? { [discriminator]: routeSchema.default }
        : {};
    const siblings = { ...defaults, ...inputs };
    const route = getDiscriminatorValue(siblings, field);
    if (
      previousInputs &&
      route === getDiscriminatorValue({ ...defaults, ...previousInputs }, field)
    )
      continue;

    const selected = inputs[name] as CredentialsMetaInput | undefined;
    if (!previousInputs && selected?.id) continue;
    const provider = getCredentialProviderFromSchema(
      siblings,
      field,
      selected?.provider,
    );
    const data =
      provider && providers && Object.hasOwn(providers, provider)
        ? providers[provider]
        : undefined;
    if (!data) {
      if (previousInputs && provider && selected?.provider !== provider)
        delete result[name];
      continue;
    }
    const { savedCredentials } = classifyCredentials(
      data.savedCredentials,
      {
        ...field,
        credentials_types: getSupportedCredentialTypes(field, route),
      },
      route,
    );
    const usable = getUsableKeys(field, savedCredentials);
    if (
      selected?.provider === provider &&
      usable.some((key) => key.id === selected.id)
    )
      continue;
    if (usable.length === 1) result[name] = toCredentialsInput(usable[0]);
    else delete result[name];
  }
  return result;
}
