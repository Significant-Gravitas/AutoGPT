import { hasRequiredScopes } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";
import {
  CredentialFieldSchema,
  findNodeCredential,
  getFittingCredentials,
  getStartingCredential,
  isOwnCredential,
  toCredentialsInput,
} from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/node-credentials";
import { getSystemCredentials } from "@/components/contextual/CredentialsInput/helpers";
import { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";

interface Args {
  properties: Record<string, CredentialFieldSchema>;
  requiredFields: string[];
  currentCredentials: Record<string, { id?: string } | undefined>;
  /** Keys the form started fields on itself; a field still on one starts again. */
  provisionalCredentials?: Record<string, string>;
  /** Fields the user set, to a key or to none; the form leaves them alone. */
  userFields?: string[];
  allProviders: CredentialsProvidersContextType;
  nodeCredentials: CredentialsMetaInput[];
  nodeCredentialsUnknown: boolean;
}

interface ScheduleArgs {
  properties: Record<string, CredentialFieldSchema>;
  requiredFields: string[];
  credentials: Record<string, CredentialsMetaInput | undefined>;
  userFields: string[];
  allProviders: CredentialsProvidersContextType | null;
  nodeCredentials: CredentialsMetaInput[];
}

/**
 * The credentials a run form starts with, for each field still empty or on a
 * key the form picked itself: the user's own key the workflow's steps agree
 * on, else the platform's key when exactly one fits.
 */
export function getDefaultRunCredentials({
  properties,
  requiredFields,
  currentCredentials,
  provisionalCredentials = {},
  userFields = [],
  allProviders,
  nodeCredentials,
  nodeCredentialsUnknown,
}: Args) {
  const defaults: Record<string, CredentialsMetaInput> = {};

  for (const [key, schema] of Object.entries(properties)) {
    if (userFields.includes(key)) continue;
    const current = currentCredentials[key];
    const isProvisional =
      current?.id !== undefined && current.id === provisionalCredentials[key];
    if (current && !isProvisional) continue;

    const credential = getStartingCredential({
      schema,
      isRequired: requiredFields.includes(key),
      nodeCredentials,
      nodeCredentialsUnknown,
      allProviders,
      findFallbackCredential: () =>
        findPlatformCredential(schema, allProviders),
    });
    if (credential) defaults[key] = credential;
  }

  return defaults;
}

/**
 * The credentials a schedule starts with. A schedule runs while no one
 * watches, and one for an expert's workflow can only use the keys the expert
 * was granted; the platform's keys need no grant. So a field the form started
 * on one of the user's own keys from the steps, and the user left alone, goes
 * back to the key it starts on without them: the platform's key for that
 * key's provider, else the only key that fits, else none. A required field
 * keeps the steps' key over none. The schedule dialog names the fields that
 * differ from the run form, and can schedule on the form's keys instead.
 */
export function getScheduleCredentials({
  properties,
  requiredFields,
  credentials,
  userFields,
  allProviders,
  nodeCredentials,
}: ScheduleArgs) {
  if (!allProviders) return credentials;
  const scheduled = { ...credentials };

  for (const [key, schema] of Object.entries(properties)) {
    const current = credentials[key];
    if (!current || userFields.includes(key)) continue;
    if (!isOwnCredential(current, allProviders)) continue;
    const nodeCredential = findNodeCredential(
      schema,
      nodeCredentials,
      allProviders,
    );
    if (nodeCredential?.id !== current.id) continue;

    // A field open to several providers keeps the step key's provider, so a
    // step set up on OpenAI isn't moved to Anthropic
    const platformCredential = findPlatformCredential(
      schema,
      allProviders,
      current.provider,
    );
    if (platformCredential) {
      scheduled[key] = platformCredential;
    } else if (
      !requiredFields.includes(key) &&
      getFittingCredentials(schema, allProviders).length > 1
    ) {
      delete scheduled[key];
    }
  }

  return scheduled;
}

// The only platform key that fits the field, from the first of its providers
// with one, or from the given provider alone
function findPlatformCredential(
  schema: CredentialFieldSchema,
  allProviders: CredentialsProvidersContextType,
  provider?: string,
) {
  const supportedTypes = schema.credentials_types ?? [];
  const providers = provider ? [provider] : (schema.credentials_provider ?? []);

  for (const providerName of providers) {
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
