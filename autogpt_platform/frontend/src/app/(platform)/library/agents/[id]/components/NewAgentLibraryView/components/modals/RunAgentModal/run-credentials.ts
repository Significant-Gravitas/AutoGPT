import { hasRequiredScopes } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";
import {
  CredentialFieldSchema,
  findNodeCredential,
  toCredentialsInput,
} from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/node-credentials";
import { getSystemCredentials } from "@/components/contextual/CredentialsInput/helpers";
import { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";

interface Args {
  properties: Record<string, CredentialFieldSchema>;
  currentCredentials: Record<string, unknown>;
  allProviders: CredentialsProvidersContextType;
  nodeCredentials: CredentialsMetaInput[];
}

/**
 * The credentials a run form starts with, for each field still empty: the
 * user's own key the workflow's steps were set to use, else the platform's
 * key when exactly one fits.
 */
export function getDefaultRunCredentials({
  properties,
  currentCredentials,
  allProviders,
  nodeCredentials,
}: Args) {
  const defaults: Record<string, CredentialsMetaInput> = {};

  for (const [key, schema] of Object.entries(properties)) {
    if (currentCredentials[key]) continue;

    const nodeCredential = findNodeCredential(
      schema,
      nodeCredentials,
      allProviders,
    );
    const credential = nodeCredential
      ? toCredentialsInput(nodeCredential)
      : findPlatformCredential(schema, allProviders);
    if (credential) defaults[key] = credential;
  }

  return defaults;
}

function findPlatformCredential(
  schema: CredentialFieldSchema,
  allProviders: CredentialsProvidersContextType,
) {
  const supportedTypes = schema.credentials_types ?? [];

  for (const providerName of schema.credentials_provider ?? []) {
    const providerData = allProviders[providerName];
    if (!providerData) continue;

    const matching = getSystemCredentials(
      providerData.savedCredentials ?? [],
    ).filter(
      (credential) =>
        supportedTypes.includes(credential.type) &&
        hasRequiredScopes(credential, schema.credentials_scopes),
    );
    if (matching.length === 1) {
      return { ...toCredentialsInput(matching[0]), provider: providerName };
    }
  }

  return undefined;
}
