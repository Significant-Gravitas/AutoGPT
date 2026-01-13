"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { toDisplayName } from "@/providers/agent-credentials/helper";
import { useContext, useMemo } from "react";
import { getSystemCredentials } from "../components/modals/CredentialsInputs/helpers";

/**
 * Hook to check if an agent is missing required SYSTEM credentials.
 * This is only used to block "New Task" buttons.
 * User credential validation is handled separately in RunAgentModal.
 */
export function useAgentMissingCredentials(
  agent: LibraryAgent | null | undefined,
) {
  const allProviders = useContext(CredentialsProvidersContext);

  const result = useMemo(() => {
    if (
      !agent ||
      !agent.id ||
      !allProviders ||
      !agent.credentials_input_schema?.properties
    ) {
      return {
        hasMissingCredentials: false,
        missingCredentials: [],
        isLoading: !allProviders || !agent,
      };
    }

    const properties = agent.credentials_input_schema.properties as Record<
      string,
      any
    >;
    const requiredCredentials = new Set(
      (agent.credentials_input_schema.required as string[]) || [],
    );

    const missingCredentials: Array<{
      key: string;
      providerDisplayName: string;
    }> = [];

    for (const [key, schema] of Object.entries(properties)) {
      const isRequired = requiredCredentials.has(key);
      if (!isRequired) continue; // Only check required credentials

      const providerNames = schema.credentials_provider || [];
      const supportedTypes = schema.credentials_types || [];
      const requiredScopes = schema.credentials_scopes;

      let hasSystemCredential = false;

      // Check if any provider has a system credential available
      for (const providerName of providerNames) {
        const providerData = allProviders[providerName];
        if (!providerData) continue;

        const systemCreds = getSystemCredentials(providerData.savedCredentials);
        const matchingSystemCreds = systemCreds.filter((cred) => {
          if (!supportedTypes.includes(cred.type)) return false;

          if (
            cred.type === "oauth2" &&
            requiredScopes &&
            requiredScopes.length > 0
          ) {
            const grantedScopes = new Set(cred.scopes || []);
            const hasAllRequiredScopes = new Set(requiredScopes).isSubsetOf(
              grantedScopes,
            );
            if (!hasAllRequiredScopes) return false;
          }

          return true;
        });

        // If there's a system credential available, it's not missing
        if (matchingSystemCreds.length > 0) {
          hasSystemCredential = true;
          break;
        }
      }

      // If no system credential available, mark as missing
      if (!hasSystemCredential) {
        const providerName = providerNames[0] || "";
        missingCredentials.push({
          key,
          providerDisplayName: toDisplayName(providerName),
        });
      }
    }

    return {
      hasMissingCredentials: missingCredentials.length > 0,
      missingCredentials,
      isLoading: false,
    };
  }, [allProviders, agent?.credentials_input_schema, agent?.id]);

  return result;
}
