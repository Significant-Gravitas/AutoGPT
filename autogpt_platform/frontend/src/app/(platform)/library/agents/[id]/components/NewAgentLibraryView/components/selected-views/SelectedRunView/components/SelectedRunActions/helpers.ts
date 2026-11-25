import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import {
  getCredentialDisplayName,
  isSystemCredential,
} from "@/components/contextual/CredentialsInput/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import type { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";
import { toDisplayName } from "@/providers/agent-credentials/helper";

// How the server refuses a key it doesn't hold for the user
const UNKNOWN_KEY_PREFIX = "Unknown credentials #";
// A provider's own key on the platform, such as "firecrawl-default"
const PLATFORM_KEY_SUFFIX = "-default";

type RunCredentials = GraphExecution["credential_inputs"];
type RunCredential = NonNullable<RunCredentials>[string];

/** A run's credential field whose key was deleted. */
export interface DeletedKey {
  field: string;
  /** The key's saved title, else its provider's name. */
  name: string;
}

/** The run form a rerun opens when it can't run on the run's keys. */
export interface RerunForm {
  deletedKeys: DeletedKey[];
  inputs: Record<string, unknown>;
  credentials: Record<string, RunCredential>;
  /** The deleted keys' fields, left for the user to pick a key for. */
  fieldsToPick: string[];
}

/**
 * The run's keys that are no longer among the user's saved ones. The
 * platform's keys are left to the server: a user can't delete them. None
 * count while the saved keys are loading, or for a provider not listed.
 */
export function getDeletedKeys(
  credentials: RunCredentials,
  allProviders: CredentialsProvidersContextType | null,
): DeletedKey[] {
  if (!allProviders) return [];
  return Object.entries(credentials ?? {}).flatMap(([field, credential]) => {
    if (!credential?.id || isPlatformKey(credential)) return [];
    if (!Object.hasOwn(allProviders, credential.provider)) return [];
    const saved = allProviders[credential.provider]?.savedCredentials;
    if (!saved || saved.some(({ id }) => id === credential.id)) return [];
    return [toDeletedKey(field, credential)];
  });
}

/** The run's own keys the server refused as unknown: deleted since the page read them. */
export function getRefusedKeys(
  error: unknown,
  credentials: RunCredentials,
): DeletedKey[] {
  const unknownIDs = new Set(
    getNodeErrorMessages(error)
      .filter((message) => message.startsWith(UNKNOWN_KEY_PREFIX))
      .map((message) => message.slice(UNKNOWN_KEY_PREFIX.length).trim()),
  );
  return Object.entries(credentials ?? {}).flatMap(([field, credential]) =>
    credential?.id &&
    unknownIDs.has(credential.id) &&
    !isPlatformKey(credential)
      ? [toDeletedKey(field, credential)]
      : [],
  );
}

/** The run's inputs and its keys that still exist; the deleted ones are left to pick. */
export function getRerunForm(
  run: Pick<GraphExecution, "inputs" | "credential_inputs">,
  deletedKeys: DeletedKey[],
): RerunForm {
  const fieldsToPick = [...new Set(deletedKeys.map(({ field }) => field))];
  return {
    deletedKeys,
    inputs: run.inputs ?? {},
    credentials: Object.fromEntries(
      Object.entries(run.credential_inputs ?? {}).filter(
        ([field]) => !fieldsToPick.includes(field),
      ),
    ),
    fieldsToPick,
  };
}

/** Names the deleted keys, as a sentence. */
export function describeDeletedKeys(deletedKeys: DeletedKey[]) {
  const names = getNames(deletedKeys);
  const list = new Intl.ListFormat("en", { type: "conjunction" }).format(names);
  return names.length === 1
    ? `A key this task used was deleted: ${list}.`
    : `Keys this task used were deleted: ${list}.`;
}

/** What the rerun form says above the task's inputs. */
export function getRerunFormNotice(deletedKeys: DeletedKey[]) {
  const pick =
    getNames(deletedKeys).length === 1
      ? "Choose another key below to run it again."
      : "Choose other keys below to run it again.";
  return `${describeDeletedKeys(deletedKeys)} ${pick}`;
}

function getNames(deletedKeys: DeletedKey[]) {
  return [...new Set(deletedKeys.map(({ name }) => name))];
}

function toDeletedKey(field: string, credential: RunCredential): DeletedKey {
  return {
    field,
    name: getCredentialDisplayName(
      { title: credential.title },
      toDisplayName(credential.provider),
    ),
  };
}

// A run's saved key doesn't say whose it is: the platform's are titled
// "Use Credits for …", or named for their provider
function isPlatformKey(credential: RunCredential) {
  return (
    isSystemCredential(credential) ||
    credential.id.endsWith(PLATFORM_KEY_SUFFIX)
  );
}

function getNodeErrorMessages(error: unknown): string[] {
  if (!(error instanceof ApiError) || !error.isGraphValidationError()) {
    return [];
  }
  return Object.values(error.response.detail.node_errors).flatMap((fields) =>
    fields && typeof fields === "object"
      ? Object.values(fields).map(String)
      : [],
  );
}
