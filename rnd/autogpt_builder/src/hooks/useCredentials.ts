import { CredentialsProviderData, CredentialsProvidersContext } from "@/components/CredentialsProvider";
import { CustomNodeData } from "@/components/CustomNode";
import {
  BlockIOCredentialsSubSchema,
} from "@/lib/autogpt-server-api";
import { useNodeId, useNodesData } from "@xyflow/react";
import { useContext } from "react";

export type CredentialsData = 
{
  provider: string;
  isApiKey: boolean;
  isOAuth2: boolean;
  schema: BlockIOCredentialsSubSchema;
  isLoading: true;
} |
CredentialsProviderData & {
  isApiKey: boolean;
  isOAuth2: boolean;
  schema: BlockIOCredentialsSubSchema;
  isLoading: false;
}

export default function useCredentials(): CredentialsData | null {
  const nodeId = useNodeId();
  const allProviders = useContext(CredentialsProvidersContext);

  if (!nodeId) {
    throw new Error("useCredentials must be within a CustomNode");
  }

  const data = useNodesData(nodeId)!.data as CustomNodeData;
  const credentials = data.inputSchema.properties.credentials as BlockIOCredentialsSubSchema

  // If block input schema doesn't have credentials, return null
  if (!credentials) {
    return null;
  }

  const provider = allProviders ? allProviders[credentials?.credentials_provider] : null;

  const isApiKey = credentials.credentials_types.includes("api_key");
  const isOAuth2 = credentials.credentials_types.includes("oauth2");

  // No provider means maybe it's still loading
  if (!provider) {
    return {
      provider: credentials.credentials_provider,
      isApiKey,
      isOAuth2,
      schema: data.inputSchema.properties.credentials as BlockIOCredentialsSubSchema,
      isLoading: true,
    };
  }

  return {
    ...provider,
    isApiKey,
    isOAuth2,
    schema: data.inputSchema.properties.credentials as BlockIOCredentialsSubSchema,
    isLoading: false,
  };
}
