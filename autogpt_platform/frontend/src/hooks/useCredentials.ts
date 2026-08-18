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
  CredentialsType,
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

export type CredentialsData = CredentialsProviderData & {
  schema: BlockIOCredentialsSubSchema;
  supportsApiKey: boolean;
  supportsOAuth2: boolean;
  supportsDeviceCode: boolean;
  supportsUserPassword: boolean;
  supportsHostScoped: boolean;
  isLoading: false;
  discriminatorValue?: string;
  upgradeableCredentials: CredentialsMetaResponse[];
  allProviderCredentials: CredentialsMetaResponse[];
};

export function getSupportedCredentialTypes(
  schema: BlockIOCredentialsSubSchema,
  discriminatorValue: string | undefined,
) {
  if (schema.discriminator_type_mapping) {
    return (
      schema.discriminator_type_mapping[discriminatorValue ?? ""] ??
      schema.credentials_types
    );
  }
  return schema.credentials_types;
}

/**
 * Maps a block's accepted credential types to the connect methods its UI
 * should offer.
 *
 * These are usually the same list, but a device-code grant produces an
 * ordinary OAuth2 credential — so such a block accepts `oauth2` (otherwise
 * saved credentials stop matching) while `device_code` is what the user must
 * actually go through. Letting `device_code` shadow `oauth2` keeps the UI
 * from offering an authorization-code redirect the provider has no client
 * secret for, which is what made these blocks unconnectable.
 */
export function deriveAuthMethods(supportedTypes: readonly CredentialsType[]) {
  const authMethods = getConnectableCredentialTypes(supportedTypes);
  return {
    supportsApiKey: authMethods.includes("api_key"),
    supportsDeviceCode: authMethods.includes("device_code"),
    supportsOAuth2: authMethods.includes("oauth2"),
    supportsUserPassword: authMethods.includes("user_password"),
    supportsHostScoped: authMethods.includes("host_scoped"),
  };
}

export function getConnectableCredentialTypes(
  supportedTypes: readonly CredentialsType[],
) {
  const usesDeviceAuth = supportedTypes.includes("device_code");
  return supportedTypes.filter((type) => type !== "oauth2" || !usesDeviceAuth);
}

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
  } else if (
    credsInputSchema.discriminator &&
    credsInputSchema.discriminator_mapping
  ) {
    // A single-provider field that still declares a mapping is saying some of
    // its discriminator values need no credential at all (AutoPilot's
    // `platform` transport). Returning the lone provider regardless would ask
    // for a credential the selected option will never use.
    if (!discriminatedProvider) return null;
    providerName = discriminatedProvider;
  } else {
    providerName = credsInputSchema.credentials_provider[0];
  }
  const provider = allProviders ? allProviders[providerName] : null;

  // If block input schema doesn't have credentials, return null
  if (!credsInputSchema) {
    return null;
  }

  const supportedTypes = getSupportedCredentialTypes(
    credsInputSchema,
    discriminatorValue,
  );
  const effectiveSchema = {
    ...credsInputSchema,
    credentials_types: supportedTypes,
  };
  const {
    supportsApiKey,
    supportsDeviceCode,
    supportsOAuth2,
    supportsUserPassword,
    supportsHostScoped,
  } = deriveAuthMethods(supportedTypes);

  // No provider means maybe it's still loading
  if (!provider) {
    return null;
  }

  const { savedCredentials, upgradeableCredentials } = classifyCredentials(
    provider.savedCredentials,
    effectiveSchema,
    discriminatorValue,
  );

  return {
    ...provider,
    allProviderCredentials: provider.savedCredentials,
    provider: providerName,
    schema: effectiveSchema,
    supportsApiKey,
    supportsOAuth2,
    supportsDeviceCode,
    supportsUserPassword,
    supportsHostScoped,
    savedCredentials,
    upgradeableCredentials,
    discriminatorValue,
    isLoading: false,
  };
}
