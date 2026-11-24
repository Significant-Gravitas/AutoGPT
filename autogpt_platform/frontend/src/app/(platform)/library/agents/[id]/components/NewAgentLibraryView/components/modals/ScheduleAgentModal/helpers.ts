import { isOwnCredential } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/node-credentials";
import { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";
import { toDisplayName } from "@/providers/agent-credentials/helper";

type Credentials = Record<string, CredentialsMetaInput | undefined>;

interface Args {
  scheduleCredentials: Credentials;
  formCredentials?: Credentials;
  allProviders: CredentialsProvidersContextType | null;
}

/** A run form field whose key the schedule doesn't run on. */
export interface ReplacedKey {
  field: string;
  /** The field, the key the schedule uses, and the one the form shows. */
  text: string;
}

export function getReplacedKeys({
  scheduleCredentials,
  formCredentials = {},
  allProviders,
}: Args): ReplacedKey[] {
  return Object.entries(formCredentials).flatMap(([field, shown]) => {
    const used = scheduleCredentials[field];
    if (!shown?.id || used?.id === shown.id) return [];

    const label = `${toDisplayName(shown.provider)} credentials`;
    const shownName = shown.title || shown.id;
    if (!used?.id) {
      return {
        field,
        text: `${label}: no key instead of ${shownName}, so the steps that use it are skipped`,
      };
    }
    const usedName = isOwnCredential(used, allProviders)
      ? used.title || used.id
      : "the platform's key";
    return { field, text: `${label}: ${usedName} instead of ${shownName}` };
  });
}
