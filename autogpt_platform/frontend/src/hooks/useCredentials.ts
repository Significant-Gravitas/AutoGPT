import { useContext } from "react";
import { CustomNodeData } from "@/components/CustomNode";
import {
  BlockIOCredentialsSubSchema,
  CredentialsProviderName,
} from "@/lib/autogpt-server-api";
import { Node, useNodeId, useNodesData } from "@xyflow/react";
import {
  CredentialsProviderData,
  CredentialsProvidersContext,
} from "@/components/integrations/credentials-provider";
import { getValue } from "@/lib/utils";

export type CredentialsData =
  | {
      provider: string;
      schema: BlockIOCredentialsSubSchema;
      supportsApiKey: boolean;
      supportsOAuth2: boolean;
      isLoading: true;
    }
  | (CredentialsProviderData & {
      schema: BlockIOCredentialsSubSchema;
      supportsApiKey: boolean;
      supportsOAuth2: boolean;
      isLoading: false;
    });

export default function useCredentials(): CredentialsData | null {
  const nodeId = useNodeId();
  const allProviders = useContext(CredentialsProvidersContext);

  if (!nodeId) {
    throw new Error("useCredentials must be within a CustomNode");
  }

  const data = useNodesData<Node<CustomNodeData>>(nodeId)!.data;
  const credentialsSchema = data.inputSchema.properties
    .credentials as BlockIOCredentialsSubSchema;

  const discriminatorValue: CredentialsProviderName | null =
    (credentialsSchema.discriminator &&
      credentialsSchema.discriminator_mapping![
        getValue(credentialsSchema.discriminator, data.hardcodedValues)
      ]) ||
    null;

  let providerName: CredentialsProviderName;
  if (credentialsSchema.credentials_provider.length > 1) {
    if (!credentialsSchema.discriminator) {
      throw new Error(
        "Multi-provider credential input requires discriminator!",
      );
    }
    if (!discriminatorValue) {
      return null;
    }
    providerName = discriminatorValue;
  } else {
    providerName = credentialsSchema.credentials_provider[0];
  }
  const provider = allProviders ? allProviders[providerName] : null;

  // If block input schema doesn't have credentials, return null
  if (!credentialsSchema) {
    return null;
  }

  const supportsApiKey =
    credentialsSchema.credentials_types.includes("api_key");
  const supportsOAuth2 = credentialsSchema.credentials_types.includes("oauth2");

  // No provider means maybe it's still loading
  if (!provider) {
    // return {
    //   provider: credentialsSchema.credentials_provider,
    //   schema: credentialsSchema,
    //   supportsApiKey,
    //   supportsOAuth2,
    //   isLoading: true,
    // };
    return null;
  }

  // Filter by OAuth credentials that have sufficient scopes for this block
  const requiredScopes = credentialsSchema.credentials_scopes;
  const savedOAuthCredentials = requiredScopes
    ? provider.savedOAuthCredentials.filter((c) =>
        new Set(c.scopes).isSupersetOf(new Set(requiredScopes)),
      )
    : provider.savedOAuthCredentials;

  return {
    ...provider,
    provider: providerName,
    schema: credentialsSchema,
    supportsApiKey,
    supportsOAuth2,
    savedOAuthCredentials,
    isLoading: false,
  };
}
