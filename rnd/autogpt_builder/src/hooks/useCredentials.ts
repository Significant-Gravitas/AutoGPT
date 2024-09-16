import { CustomNodeData } from "@/components/CustomNode";
import AutoGPTServerAPI, {
  BlockIOCredentialsSubSchema,
  CredentialsResponse,
} from "@/lib/autogpt-server-api";
import { useNodeId, useNodesData } from "@xyflow/react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

export default function useCredentials() {
  const nodeId = useNodeId();

  if (!nodeId) {
    throw new Error("useCredentials must be within a CustomNode");
  }

  const router = useRouter();
  const [savedApiKeys, setSavedApiKeys] = useState<CredentialsResponse[]>([]);
  const [savedOAuthCredentials, setSavedOAuthCredentials] = useState<
    CredentialsResponse[]
  >([]);

  const data = useNodesData(nodeId)!.data as CustomNodeData;
  const credentials = data.inputSchema.properties.credentials as BlockIOCredentialsSubSchema

  const api = useMemo(() => new AutoGPTServerAPI(process.env.NEXT_PUBLIC_AGPT_SERVER_URL!), []);

  useEffect(() => {
    if (!credentials) {
      return;
    }

    api
      .listOAuthCredentials(credentials.credentials_provider)
      .then((response) => {
        const { oauthCreds, apiKeys } = response.reduce<{
          oauthCreds: CredentialsResponse[];
          apiKeys: CredentialsResponse[];
        }>(
          (acc, cred) => {
            if (cred.credentials_type === "oauth2") {
              acc.oauthCreds.push(cred as CredentialsResponse);
            } else if (cred.credentials_type === "api_key") {
              acc.apiKeys.push(cred as CredentialsResponse);
            }
            return acc;
          },
          { oauthCreds: [], apiKeys: [] },
        );

        setSavedOAuthCredentials(oauthCreds);
        setSavedApiKeys(apiKeys);
      });
  }, [api]);

  const oAuthLogin = useCallback(
    async (scopes: string) => {
      const { login_url } = await api.oAuthLogin(
        credentials.credentials_provider,
        scopes,
      );
      router.push(login_url);
    },
    [api, credentials],
  );

  // If block input schema doesn't have credentials, return null
  if (!credentials) {
    return null;
  }

  const isLoading = savedApiKeys === null;

  const providerName = {
    github: "GitHub",
    google: "Google",
    notion: "Notion",
  };

  const isApiKey = credentials.credentials_types.includes("api_key");
  const isOAuth2 = credentials.credentials_types.includes("oauth2");

  return {
    provider: credentials.credentials_provider,
    providerName: providerName[credentials.credentials_provider],
    savedApiKeys,
    savedOAuthCredentials,
    isApiKey,
    isOAuth2,
    schema: data.inputSchema.properties.credentials as BlockIOCredentialsSubSchema,
    oAuthLogin,
    isLoading,
  };
}
