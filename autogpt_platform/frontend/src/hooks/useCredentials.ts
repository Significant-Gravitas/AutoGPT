import { useContext } from "react";
import { CustomNodeData } from "@/components/CustomNode";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { Node, useNodeId, useNodesData } from "@xyflow/react";
import {
  CredentialsProviderData,
  CredentialsProvidersContext,
} from "@/components/integrations/credentials-provider";

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

  // If block input schema doesn't have credentials, return null
  if (!credentialsSchema) {
    return null;
  }

  const provider = allProviders
    ? allProviders[credentialsSchema?.credentials_provider]
    : null;

  const supportsApiKey =
    credentialsSchema.credentials_types.includes("api_key");
  const supportsOAuth2 = credentialsSchema.credentials_types.includes("oauth2");

  // No provider means maybe it's still loading
  if (!provider) {
    return {
      provider: credentialsSchema.credentials_provider,
      schema: credentialsSchema,
      supportsApiKey,
      supportsOAuth2,
      isLoading: true,
    };
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
    schema: credentialsSchema,
    supportsApiKey,
    supportsOAuth2,
    savedOAuthCredentials,
    isLoading: false,
  };
}
