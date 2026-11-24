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

interface GraphSteps {
  nodes?: Array<{ input_default?: Record<string, unknown> }>;
}

interface StartingCredentialArgs {
  schema: CredentialFieldSchema;
  nodeCredentials: CredentialsMetaInput[];
  /** The steps' keys couldn't be read, so which key they use is unknown. */
  nodeCredentialsUnknown?: boolean;
  allProviders: CredentialsProvidersContextType | null;
  /** The key the field starts on when the steps don't decide, e.g. the platform's. */
  findFallbackCredential: () => CredentialsMetaInput | undefined;
}

/** The credentials picked on a workflow's steps, read from each step's saved inputs. */
export function getSavedNodeCredentials(
  nodeInputs: Array<Record<string, unknown> | null | undefined>,
): CredentialsMetaInput[] {
  return nodeInputs.flatMap((inputs) =>
    Object.values(inputs ?? {}).filter(isCredentialsMetaInput),
  );
}

/** The inputs saved on these graphs' steps, e.g. a workflow's and its sub-workflows'. */
export function getStepInputs(graphs: GraphSteps[]) {
  return graphs.flatMap((graph) =>
    (graph.nodes ?? []).map((node) => node.input_default),
  );
}

/** Tells two sets of step keys apart, whatever order the steps come in. */
export function getNodeCredentialsKey(nodeCredentials: CredentialsMetaInput[]) {
  return nodeCredentials
    .map((credential) => `${credential.provider}/${credential.id}`)
    .sort()
    .join(",");
}

/**
 * The key a run field starts on: the user's own key the workflow's steps
 * agree on, else the field's fallback. While the steps' keys are unknown, a
 * field the user has a key of their own for starts on none, so the platform's
 * key can't silently replace a step's: the user picks.
 */
export function getStartingCredential({
  schema,
  nodeCredentials,
  nodeCredentialsUnknown = false,
  allProviders,
  findFallbackCredential,
}: StartingCredentialArgs) {
  const nodeCredential = findNodeCredential(
    schema,
    nodeCredentials,
    allProviders,
  );
  if (nodeCredential) return toCredentialsInput(nodeCredential);
  if (awaitsUserChoice(schema, nodeCredentialsUnknown, allProviders)) {
    return undefined;
  }
  return findFallbackCredential();
}

/** Whether a field's key is left for the user to pick, as the steps' keys are unknown. */
export function awaitsUserChoice(
  schema: CredentialFieldSchema,
  nodeCredentialsUnknown: boolean,
  allProviders: CredentialsProvidersContextType | null,
) {
  return nodeCredentialsUnknown && hasOwnCredentialFor([schema], allProviders);
}

/**
 * The user's own key that every step under this field was set to use. Each
 * step's saved key counts, the platform's included; a step with no key, or
 * with a key that no longer exists, doesn't. Steps set to different keys give
 * no answer, and so do steps set to the platform's key, which is what the
 * field falls back to anyway.
 */
export function findNodeCredential(
  schema: CredentialFieldSchema,
  nodeCredentials: CredentialsMetaInput[],
  allProviders: CredentialsProvidersContextType | null,
) {
  const choices = new Map<string, CredentialsMetaResponse>();
  for (const nodeCredential of nodeCredentials) {
    const saved = findSavedCredential(nodeCredential, allProviders);
    if (saved && isUnderField(saved, schema)) choices.set(saved.id, saved);
  }

  if (choices.size !== 1) return undefined;
  const [choice] = choices.values();
  if (isSystemCredential(choice)) return undefined;
  return hasRequiredScopes(choice, schema.credentials_scopes)
    ? choice
    : undefined;
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

// Whether a step's key sits under this field, which the run fills for every
// step it covers. Scopes don't decide it: the field asks for all its steps'.
function isUnderField(
  credential: CredentialsMetaResponse,
  schema: CredentialFieldSchema,
) {
  const credentialTypes = schema.credentials_types ?? [];
  return (
    (schema.credentials_provider ?? []).includes(credential.provider) &&
    (credentialTypes.length === 0 ||
      credentialTypes.includes(credential.type)) &&
    matchesDiscriminatorValues(credential, schema.discriminator_values)
  );
}

function findSavedCredential(
  credential: CredentialsMetaInput,
  allProviders: CredentialsProvidersContextType | null,
) {
  return allProviders?.[credential.provider]?.savedCredentials.find(
    (saved) => saved.id === credential.id,
  );
}
