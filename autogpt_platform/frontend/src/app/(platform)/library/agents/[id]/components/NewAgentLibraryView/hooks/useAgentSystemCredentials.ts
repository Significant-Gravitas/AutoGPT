"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { CredentialsMetaResponse } from "@/lib/autogpt-server-api/types";
import {
  CredentialsProviderData,
  CredentialsProvidersContext,
} from "@/providers/agent-credentials/credentials-provider";
import { toDisplayName } from "@/providers/agent-credentials/helper";
import { useContext, useMemo } from "react";
import {
  filterSystemCredentials,
  getSystemCredentials,
} from "../components/modals/CredentialsInputs/helpers";

interface SystemCredentialInfo {
  key: string;
  provider: string;
  schema: any;
  credential: CredentialsMetaResponse;
}

interface MissingCredentialInfo {
  key: string;
  provider: string;
  providerDisplayName: string;
}

interface UseAgentSystemCredentialsResult {
  hasSystemCredentials: boolean;
  systemCredentials: SystemCredentialInfo[];
  hasMissingSystemCredentials: boolean;
  missingSystemCredentials: MissingCredentialInfo[];
  isLoading: boolean;
}

export function useAgentSystemCredentials(
  agent: LibraryAgent,
): UseAgentSystemCredentialsResult {
  const allProviders = useContext(CredentialsProvidersContext);

  const result = useMemo(() => {
    const empty = {
      hasSystemCredentials: false,
      systemCredentials: [],
      hasMissingSystemCredentials: false,
      missingSystemCredentials: [],
      isLoading: false,
    };

    if (!agent.credentials_input_schema?.properties) return empty;

    if (!allProviders) return { ...empty, isLoading: true };

    const properties = agent.credentials_input_schema.properties as Record<
      string,
      any
    >;
    const requiredCredentials = new Set(
      (agent.credentials_input_schema.required as string[]) || [],
    );
    const systemCredentials: SystemCredentialInfo[] = [];
    const missingSystemCredentials: MissingCredentialInfo[] = [];

    for (const [key, schema] of Object.entries(properties)) {
      const providerNames = schema.credentials_provider || [];
      const isRequired = requiredCredentials.has(key);
      const supportedTypes = schema.credentials_types || [];

      for (const providerName of providerNames) {
        const providerData: CredentialsProviderData | undefined =
          allProviders[providerName];

        if (!providerData) {
          // Provider not loaded yet - don't mark as missing, wait for load
          continue;
        }

        // Check for system credentials - now backend always returns them with is_system: true
        const systemCreds = getSystemCredentials(providerData.savedCredentials);
        const userCreds = filterSystemCredentials(
          providerData.savedCredentials,
        );

        const matchingSystemCreds = systemCreds.filter((cred) =>
          supportedTypes.includes(cred.type),
        );
        const matchingUserCreds = userCreds.filter((cred) =>
          supportedTypes.includes(cred.type),
        );

        // Add system credentials if they exist (even if not configured, backend returns them)
        for (const cred of matchingSystemCreds) {
          systemCredentials.push({
            key,
            provider: providerName,
            schema,
            credential: cred,
          });
        }

        // Only mark as missing if it's required AND there are NO credentials available
        // (neither system nor user). This is a true "missing" state.
        // Note: We don't block based on this anymore since the run modal
        // has its own validation (allRequiredInputsAreSet)
        if (
          isRequired &&
          matchingSystemCreds.length === 0 &&
          matchingUserCreds.length === 0
        ) {
          missingSystemCredentials.push({
            key,
            provider: providerName,
            providerDisplayName: toDisplayName(providerName),
          });
        }
      }
    }

    return {
      hasSystemCredentials: systemCredentials.length > 0,
      systemCredentials,
      hasMissingSystemCredentials: missingSystemCredentials.length > 0,
      missingSystemCredentials,
      isLoading: false,
    };
  }, [agent.credentials_input_schema, allProviders]);

  return result;
}
