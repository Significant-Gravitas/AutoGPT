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
    });

export default function useCredentials(
  credsInputSchema: BlockIOCredentialsSubSchema,
  nodeInputValues?: Record<string, any>,
): CredentialsData | null {
  const allProviders = useContext(CredentialsProvidersContext);

  // If block input schema doesn't have credentials, return null
  if (!credsInputSchema) {
    return null;
  }

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
  
  // Handle cases where credentials_provider might be undefined or not an array
  const credentialsProviders = Array.isArray(credsInputSchema.credentials_provider) 
    ? credsInputSchema.credentials_provider 
    : credsInputSchema.credentials_provider 
      ? [credsInputSchema.credentials_provider]
      : [];
  
  if (credentialsProviders.length === 0) {
    console.warn("No credentials provider specified in schema");
    return null;
  }
  
  if (credentialsProviders.length > 1) {
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
    providerName = credentialsProviders[0];
  }
  const provider = allProviders ? allProviders[providerName] : null;

  // Safely handle credentials_types which might be undefined
  const credentialsTypes = credsInputSchema.credentials_types || [];
  const supportsApiKey = credentialsTypes.includes("api_key");
  const supportsOAuth2 = credentialsTypes.includes("oauth2");
  const supportsUserPassword = credentialsTypes.includes("user_password");
  const supportsHostScoped = credentialsTypes.includes("host_scoped");

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
    // First, check if the credential type is supported by this block
    const supportedTypes = credsInputSchema.credentials_types;
    if (!supportedTypes.includes(c.type)) {
      return false;
    }

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
      return discriminatorValue && getHostFromUrl(discriminatorValue) == c.host;
    }

    // Include all other credential types that passed the type check
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
    discriminatorValue,
    isLoading: false,
  };
}
