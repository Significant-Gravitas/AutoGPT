import { useContext } from "react";
import { getValue } from "@/lib/utils";

import {
  CredentialsProviderData,
  CredentialsProvidersContext,
} from "@/components/integrations/credentials-provider";
import {
  BlockIOCredentialsSubSchema,
  CredentialsProviderName,
} from "@/lib/autogpt-server-api";
import { getHostFromUrl } from "@/lib/utils/url";

export type CredentialsData =
  | {
      provider: string;
      schema: BlockIOCredentialsSubSchema;
      supportsApiKey: boolean;
      supportsOAuth2: boolean;
      supportsUserPassword: boolean;
      supportsHostScoped: boolean;
      isLoading: true;
    }
  | (CredentialsProviderData & {
      schema: BlockIOCredentialsSubSchema;
      supportsApiKey: boolean;
      supportsOAuth2: boolean;
      supportsUserPassword: boolean;
      supportsHostScoped: boolean;
      isLoading: false;
    });

export default function useCredentials(
  credsInputSchema: BlockIOCredentialsSubSchema,
  nodeInputValues?: Record<string, any>,
): CredentialsData | null {
  const allProviders = useContext(CredentialsProvidersContext);

  const discriminatorValue: CredentialsProviderName | null = (() => {
    if (
      !credsInputSchema.discriminator ||
      !credsInputSchema.discriminator_mapping
    ) {
      return null;
    }

    const discriminatorKey = getValue(
      credsInputSchema.discriminator,
      nodeInputValues,
    );
    if (discriminatorKey === undefined) {
      return null;
    }

    return credsInputSchema.discriminator_mapping[discriminatorKey] || null;
  })();

  let providerName: CredentialsProviderName;
  if (credsInputSchema.credentials_provider.length > 1) {
    if (!credsInputSchema.discriminator) {
      throw new Error(
        "Multi-provider credential input requires discriminator!",
      );
    }
    if (!discriminatorValue) {
      console.log(
        `Missing discriminator value from '${credsInputSchema.discriminator}': ` +
          "hiding credentials input until it is set.",
      );
      return null;
    }
    providerName = discriminatorValue;
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
    // return {
    //   provider: credsInputSchema.credentials_provider,
    //   schema: credsInputSchema,
    //   supportsApiKey,
    //   supportsOAuth2,
    //   isLoading: true,
    // };
    return null;
  }

  const savedCredentials = provider.savedCredentials.filter((c) => {
    // Filter by OAuth credentials that have sufficient scopes for this block
    if (c.type === "oauth2") {
      const requiredScopes = credsInputSchema.credentials_scopes;
      return (
        !requiredScopes ||
        new Set(c.scopes).isSupersetOf(new Set(requiredScopes))
      );
    }

    // Filter host_scoped credentials by host matching
    if (c.type === "host_scoped") {
      const schemaHosts = [
        ...[credsInputSchema.discriminator_values || []],
        getValue(credsInputSchema.discriminator || "url", nodeInputValues),
      ]
        .map((v) => getHostFromUrl(v))
        .filter((h) => h);
      return schemaHosts && schemaHosts.includes(c.host || "");
    }

    // Include all other credential types
    return true;
  });

  return {
    ...provider,
    provider: providerName,
    schema: credsInputSchema,
    supportsApiKey,
    supportsOAuth2,
    supportsUserPassword,
    supportsHostScoped,
    savedCredentials,
    isLoading: false,
  };
}
