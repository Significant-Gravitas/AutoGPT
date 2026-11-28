import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { GraphModel } from "@/app/api/__generated__/models/graphModel";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { getCredentialDisplayName } from "@/components/contextual/CredentialsInput/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import type { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";
import { toDisplayName } from "@/providers/agent-credentials/helper";

export const RERUN_FORM_TITLE = "Rerun task";

// How the server refuses a key it doesn't hold for the user
export const UNKNOWN_KEY_PREFIX = "Unknown credentials #";
// A provider's own key on the platform, such as "firecrawl-default": the
// server never gives a user's key an id like it
export const SDK_KEY_SUFFIX = "-default";

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
  /** The run it reruns. */
  runId: string;
  deletedKeys: DeletedKey[];
  inputs: Record<string, unknown>;
  credentials: Record<string, RunCredential>;
  /** The deleted keys' fields, left for the user to pick a key for, each
   *  with the provider of the key it had. */
  fieldsToPick: Record<string, string>;
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
    if (!credential?.id || isPlatformKey(credential, allProviders)) return [];
    if (!Object.hasOwn(allProviders, credential.provider)) return [];
    const saved = allProviders[credential.provider]?.savedCredentials;
    if (!saved || saved.some(({ id }) => id === credential.id)) return [];
    return [toDeletedKey(field, credential)];
  });
}

/**
 * The run's own keys the server refused as unknown: deleted since the page
 * read them. None count while the saved keys are loading, as they alone mark
 * which ids are the platform's.
 */
export function getRefusedKeys(
  error: unknown,
  credentials: RunCredentials,
  allProviders: CredentialsProvidersContextType | null,
): DeletedKey[] {
  if (!allProviders) return [];
  const unknownIDs = new Set(
    getNodeErrorMessages(error)
      .filter((message) => message.startsWith(UNKNOWN_KEY_PREFIX))
      .map((message) => message.slice(UNKNOWN_KEY_PREFIX.length).trim()),
  );
  return Object.entries(credentials ?? {}).flatMap(([field, credential]) =>
    credential?.id &&
    unknownIDs.has(credential.id) &&
    !isPlatformKey(credential, allProviders)
      ? [toDeletedKey(field, credential)]
      : [],
  );
}

/** The run's inputs and its keys that still exist; the deleted ones are left to pick. */
export function getRerunForm(
  run: Pick<GraphExecution, "id" | "inputs" | "credential_inputs">,
  deletedKeys: DeletedKey[],
): RerunForm {
  const deletedFields = new Set(deletedKeys.map(({ field }) => field));
  const runCredentials = Object.entries(run.credential_inputs ?? {});
  return {
    runId: run.id,
    deletedKeys,
    inputs: run.inputs ?? {},
    credentials: Object.fromEntries(
      runCredentials.filter(([field]) => !deletedFields.has(field)),
    ),
    fieldsToPick: Object.fromEntries(
      runCredentials.flatMap(([field, credential]) =>
        deletedFields.has(field) && credential
          ? [[field, credential.provider]]
          : [],
      ),
    ),
  };
}

/**
 * The library's workflow as it was at a past version: what that version
 * asks for, the keys it takes and the checks it runs, and that version to run.
 * Its trigger fields stay the library's: a rerun runs the task's inputs, as
 * Rerun does for a task whose keys still exist, and never sets up a trigger.
 */
export function getAgentAtVersion(
  agent: LibraryAgent,
  graph: GraphModel,
  version: number,
): LibraryAgent {
  return {
    ...agent,
    graph_version: version,
    input_schema: graph.input_schema,
    credentials_input_schema: graph.credentials_input_schema,
    has_human_in_the_loop: graph.has_human_in_the_loop,
    has_sensitive_action: graph.has_sensitive_action,
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

/** What shows in place of the rerun form when the task's version can't be read. */
export function getRunVersionErrorNotice(version: number) {
  return `This task ran version ${version} of the workflow, which couldn't be loaded. Try again, or run the current version with New workflow task.`;
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

// One of the platform's keys, which a user can't delete. A run's stored key
// doesn't say whose it is, so its id decides, never its title: users name
// their keys. The saved keys mark the platform's by id, and one the page
// doesn't list may still be a provider's own SDK key.
function isPlatformKey(
  credential: RunCredential,
  allProviders: CredentialsProvidersContextType | null,
) {
  if (credential.id.endsWith(SDK_KEY_SUFFIX)) return true;
  if (!allProviders || !Object.hasOwn(allProviders, credential.provider)) {
    return false;
  }
  return (allProviders[credential.provider]?.savedCredentials ?? []).some(
    (saved) => saved.id === credential.id && saved.is_system === true,
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
