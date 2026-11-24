import {
  CredentialsMetaInput,
  CredentialsMetaResponse,
} from "@/lib/autogpt-server-api/types";
import { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";
import { isSystemCredential } from "../../helpers";
import { hasRequiredScopes, matchesDiscriminatorValues } from "./helpers";

export interface CredentialFieldSchema {
  credentials_provider?: string[];
  credentials_types?: string[];
  credentials_scopes?: string[];
  discriminator_values?: string[];
}

/** The credentials picked on a workflow's steps, read from each step's saved inputs. */
export function getSavedNodeCredentials(
  nodeInputs: Array<Record<string, unknown> | null | undefined>,
): CredentialsMetaInput[] {
  return nodeInputs.flatMap((inputs) =>
    Object.values(inputs ?? {}).filter(isCredentialsMetaInput),
  );
}

function isCredentialsMetaInput(value: unknown): value is CredentialsMetaInput {
  if (typeof value !== "object" || value === null) return false;
  const { id, provider, type } = value as Record<string, unknown>;
  return (
    typeof id === "string" &&
    id !== "" &&
    typeof provider === "string" &&
    typeof type === "string"
  );
}

/**
 * The user's own key that the workflow's steps were set to use for this
 * field. It has to still exist, belong to the user and fit the field. Steps
 * set to different keys give no answer, and a platform key saved on a step
 * gives none either: it is what the field falls back to anyway.
 */
export function findNodeCredential(
  schema: CredentialFieldSchema,
  nodeCredentials: CredentialsMetaInput[],
  allProviders: CredentialsProvidersContextType | null,
) {
  const providerNames = schema.credentials_provider ?? [];
  const credentialTypes = schema.credentials_types ?? [];
  const matches = new Map<string, CredentialsMetaResponse>();

  for (const nodeCredential of nodeCredentials) {
    if (!providerNames.includes(nodeCredential.provider)) continue;
    const saved = findSavedCredential(nodeCredential, allProviders);
    if (!saved || isSystemCredential(saved)) continue;
    if (credentialTypes.length > 0 && !credentialTypes.includes(saved.type))
      continue;
    if (!hasRequiredScopes(saved, schema.credentials_scopes)) continue;
    if (!matchesDiscriminatorValues(saved, schema.discriminator_values))
      continue;
    matches.set(saved.id, saved);
  }

  return matches.size === 1 ? [...matches.values()][0] : undefined;
}

/** Whether a picked credential is one of the user's own, not a platform key. */
export function isOwnCredential(
  selected: CredentialsMetaInput | undefined,
  allProviders: CredentialsProvidersContextType | null,
) {
  if (!selected?.id) return false;
  const saved = findSavedCredential(selected, allProviders);
  return saved !== undefined && !isSystemCredential(saved);
}

/** Whether the user has a key of their own for any of these fields' providers. */
export function hasOwnCredentialFor(
  schemas: CredentialFieldSchema[],
  allProviders: CredentialsProvidersContextType | null,
) {
  return schemas.some((schema) =>
    (schema.credentials_provider ?? []).some((providerName) =>
      (allProviders?.[providerName]?.savedCredentials ?? []).some(
        (credential) => !isSystemCredential(credential),
      ),
    ),
  );
}

export function toCredentialsInput(
  credential: CredentialsMetaResponse,
): CredentialsMetaInput {
  return {
    id: credential.id,
    provider: credential.provider,
    type: credential.type,
    title: credential.title,
  };
}

function findSavedCredential(
  credential: CredentialsMetaInput,
  allProviders: CredentialsProvidersContextType | null,
) {
  return allProviders?.[credential.provider]?.savedCredentials.find(
    (saved) => saved.id === credential.id,
  );
}
