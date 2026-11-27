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

interface UserChoiceArgs {
  schema: CredentialFieldSchema;
  /** Whether the run needs the field filled. */
  isRequired: boolean;
  /** The steps' keys couldn't be read, so which key they use is unknown. */
  nodeCredentialsUnknown?: boolean;
  allProviders: CredentialsProvidersContextType | null;
}

interface StartingCredentialArgs extends UserChoiceArgs {
  nodeCredentials: CredentialsMetaInput[];
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
 * required field the user has a key of their own for starts on none, so the
 * platform's key can't silently replace a step's: the user picks.
 */
export function getStartingCredential({
  schema,
  isRequired,
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
  if (
    awaitsUserChoice({
      schema,
      isRequired,
      nodeCredentialsUnknown,
      allProviders,
    })
  ) {
    return undefined;
  }
  return findFallbackCredential();
}

/**
 * Whether a field's key is left for the user to pick: the steps' keys are
 * unknown, the run needs the field, and the user has a key of their own that
 * fits it. An optional field starts on the platform's key.
 */
export function awaitsUserChoice({
  schema,
  isRequired,
  nodeCredentialsUnknown = false,
  allProviders,
}: UserChoiceArgs) {
  return (
    nodeCredentialsUnknown &&
    isRequired &&
    hasOwnCredentialFor([schema], allProviders)
  );
}

/**
 * The key every step under this field was set to use. Each step's saved key
 * counts, the platform's included; a step with no key, or with a key that no
 * longer exists, doesn't. Steps set to different keys give no answer. So do
 * steps set to the platform's key on a field with one provider, which falls
 * back to that key anyway; a field open to several providers falls back to
 * the first one's, so there the platform key the steps agree on is named.
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
  const hasOneProvider = (schema.credentials_provider ?? []).length <= 1;
  if (isSystemCredential(choice) && hasOneProvider) return undefined;
  return hasRequiredScopes(choice, schema.credentials_scopes)
    ? choice
    : undefined;
}

/** The saved keys, the platform's included, that fit a field: provider, type, host and scopes. */
export function getFittingCredentials(
  schema: CredentialFieldSchema,
  allProviders: CredentialsProvidersContextType | null,
) {
  return (schema.credentials_provider ?? []).flatMap((providerName) =>
    (allProviders?.[providerName]?.savedCredentials ?? []).filter(
      (credential) =>
        isUnderField(credential, schema) &&
        hasRequiredScopes(credential, schema.credentials_scopes),
    ),
  );
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

/** Whether the user has a key of their own that fits any of these fields. */
export function hasOwnCredentialFor(
  schemas: CredentialFieldSchema[],
  allProviders: CredentialsProvidersContextType | null,
) {
  return schemas.some(
    (schema) => findOwnCredentialProvider(schema, allProviders) !== undefined,
  );
}

/** The first of a field's providers the user has a key of their own for that fits the field. */
export function findOwnCredentialProvider(
  schema: CredentialFieldSchema,
  allProviders: CredentialsProvidersContextType | null,
) {
  return (schema.credentials_provider ?? []).find((providerName) =>
    (allProviders?.[providerName]?.savedCredentials ?? []).some(
      (credential) =>
        !isSystemCredential(credential) && isUnderField(credential, schema),
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

// Whether a key sits under this field, which the run fills for every step it
// covers: its provider, type and host fit. Scopes don't decide it: the field
// asks for all its steps'.
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
  // A step's saved inputs are workflow data: a provider name outside the
  // list, such as "constructor", must not reach an inherited property.
  if (!allProviders || !Object.hasOwn(allProviders, credential.provider))
    return undefined;
  return allProviders[credential.provider]?.savedCredentials.find(
    (saved) => saved.id === credential.id,
  );
}
