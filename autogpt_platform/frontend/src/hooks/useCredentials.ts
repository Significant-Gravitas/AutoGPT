import { useContext } from "react";
import { getValue } from "@/lib/utils";

import {
  CredentialsProviderData,
  CredentialsProvidersContext,
} from "@/providers/agent-credentials/credentials-provider";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaResponse,
  CredentialsProviderName,
} from "@/lib/autogpt-server-api";
import { getHostFromUrl } from "@/lib/utils/url";

export function classifyCredentials(
  allSaved: readonly CredentialsMetaResponse[],
  credsInputSchema: BlockIOCredentialsSubSchema,
  discriminatorValue: string | undefined,
): {
  savedCredentials: CredentialsMetaResponse[];
  upgradeableCredentials: CredentialsMetaResponse[];
} {
  const savedCredentials: CredentialsMetaResponse[] = [];
  const upgradeableCredentials: CredentialsMetaResponse[] = [];
  const supportedTypes = credsInputSchema.credentials_types;

  for (const c of allSaved) {
    if (!supportedTypes.includes(c.type)) continue;

    // MCP OAuth2 credentials filter by server URL — not upgradeable
    if (c.type === "oauth2" && c.provider === "mcp") {
      if (discriminatorValue != null && c.host === discriminatorValue) {
        savedCredentials.push(c);
      }
      continue;
    }

    if (c.type === "oauth2") {
      const requiredScopes = credsInputSchema.credentials_scopes;
      // Set.prototype.isSupersetOf is ES2025 and this project targets
      // ES2022 — fall back to an array every() check so the picker's
      // scope filter runs cleanly on current Node/browser baselines.
      const credScopes = new Set(c.scopes);
      const hasAllScopes =
        !requiredScopes || requiredScopes.every((s) => credScopes.has(s));
      if (hasAllScopes) {
        savedCredentials.push(c);
      } else {
        upgradeableCredentials.push(c);
      }
      continue;
    }

    if (c.type === "host_scoped") {
      if (discriminatorValue && getHostFromUrl(discriminatorValue) == c.host) {
        savedCredentials.push(c);
      }
      continue;
    }

    savedCredentials.push(c);
  }

  return { savedCredentials, upgradeableCredentials };
}

export type CredentialsData =
  | {
      provider: string;
      schema: BlockIOCredentialsSubSchema;
      supportsApiKey: boolean;
      supportsOAuth2: boolean;
      supportsUserPassword: boolean;
      supportsHostScoped: boolean;
      isLoading: true;
      discriminatorValue?: string;
    }
  | (CredentialsProviderData & {
      schema: BlockIOCredentialsSubSchema;
      supportsApiKey: boolean;
      supportsOAuth2: boolean;
      supportsUserPassword: boolean;
      supportsHostScoped: boolean;
      isLoading: false;
      discriminatorValue?: string;
      upgradeableCredentials: CredentialsMetaResponse[];
    });

export default function useCredentials(
  credsInputSchema: BlockIOCredentialsSubSchema,
  nodeInputValues?: Record<string, any>,
): CredentialsData | null {
  const allProviders = useContext(CredentialsProvidersContext);

  const discriminatorValue = [
    credsInputSchema.discriminator
      ? getValue(credsInputSchema.discriminator, nodeInputValues)
      : null,
    ...(credsInputSchema.discriminator_values || []),
  ].find(Boolean);

  const discriminatedProvider = credsInputSchema.discriminator_mapping
    ? credsInputSchema.discriminator_mapping[discriminatorValue]
    : null;

  let providerName: CredentialsProviderName;
  if (credsInputSchema.credentials_provider.length > 1) {
    if (!credsInputSchema.discriminator) {
      throw new Error(
        "Multi-provider credential input requires discriminator!",
      );
    }
    if (!discriminatedProvider) {
      console.warn(
        `Missing discriminator value from '${credsInputSchema.discriminator}': ` +
          "hiding credentials input until it is set.",
      );
      return null;
    }
    providerName = discriminatedProvider;
  } else {
    providerName = credsInputSchema.credentials_provider[0];
  }
  const provider = allProviders ? allProviders[providerName] : null;

  // If block input schema doesn't have credentials, return null
  if (!credsInputSchema) {
    return null;
  }

  const supportsApiKey = credsInputSchema.credentials_types.includes("api_key");
  const supportsOAuth2 = credsInputSchema.credentials_types.includes("oauth2");
  const supportsUserPassword =
    credsInputSchema.credentials_types.includes("user_password");
  const supportsHostScoped =
    credsInputSchema.credentials_types.includes("host_scoped");

  // No provider means maybe it's still loading
  if (!provider) {
    return null;
  }

  const { savedCredentials, upgradeableCredentials } = classifyCredentials(
    provider.savedCredentials,
    credsInputSchema,
    discriminatorValue,
  );

  return {
    ...provider,
    provider: providerName,
    schema: credsInputSchema,
    supportsApiKey,
    supportsOAuth2,
    supportsUserPassword,
    supportsHostScoped,
    savedCredentials,
    upgradeableCredentials,
    discriminatorValue,
    isLoading: false,
  };
}
