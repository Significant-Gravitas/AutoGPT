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

export type CredentialsData =
  | {
      provider: string;
      schema: BlockIOCredentialsSubSchema;
      supportsApiKey: boolean;
      supportsOAuth2: boolean;
      supportsUserPassword: boolean;
      isLoading: true;
    }
  | (CredentialsProviderData & {
      schema: BlockIOCredentialsSubSchema;
      supportsApiKey: boolean;
      supportsOAuth2: boolean;
      supportsUserPassword: boolean;
      isLoading: false;
    });

export default function useCredentials(
  credsInputSchema: BlockIOCredentialsSubSchema,
  nodeInputValues?: Record<string, any>,
): CredentialsData | null {
  const allProviders = useContext(CredentialsProvidersContext);

  const discriminatorValue: CredentialsProviderName | null =
    (credsInputSchema.discriminator &&
      credsInputSchema.discriminator_mapping![
        getValue(credsInputSchema.discriminator, nodeInputValues)
      ]) ||
    null;

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

  // Filter by OAuth credentials that have sufficient scopes for this block
  const requiredScopes = credsInputSchema.credentials_scopes;
  const savedCredentials = requiredScopes
    ? provider.savedCredentials.filter(
        (c) =>
          c.type != "oauth2" ||
          new Set(c.scopes).isSupersetOf(new Set(requiredScopes)),
      )
    : provider.savedCredentials;

  return {
    ...provider,
    provider: providerName,
    schema: credsInputSchema,
    supportsApiKey,
    supportsOAuth2,
    supportsUserPassword,
    savedCredentials,
    isLoading: false,
  };
}
