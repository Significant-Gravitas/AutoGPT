import AutoGPTServerAPI, {
  APIKeyCredentials,
  CredentialsDeleteResponse,
  CredentialsMetaResponse,
  CredentialsProviderName,
  PROVIDER_NAMES,
} from "@/lib/autogpt-server-api";
import {
  createContext,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

// Get keys from CredentialsProviderName type
const CREDENTIALS_PROVIDER_NAMES = Object.values(
  PROVIDER_NAMES,
) as CredentialsProviderName[];

// --8<-- [start:CredentialsProviderNames]
const providerDisplayNames: Record<CredentialsProviderName, string> = {
  discord: "Discord",
  d_id: "D-ID",
  github: "GitHub",
  google: "Google",
  google_maps: "Google Maps",
  ideogram: "Ideogram",
  jina: "Jina",
  medium: "Medium",
  llm: "LLM",
  notion: "Notion",
  openai: "OpenAI",
  openweathermap: "OpenWeatherMap",
  pinecone: "Pinecone",
  replicate: "Replicate",
  revid: "Rev.ID",
  unreal_speech: "Unreal Speech",
} as const;
// --8<-- [end:CredentialsProviderNames]

type APIKeyCredentialsCreatable = Omit<
  APIKeyCredentials,
  "id" | "provider" | "type"
>;

export type CredentialsProviderData = {
  provider: CredentialsProviderName;
  providerName: string;
  savedApiKeys: CredentialsMetaResponse[];
  savedOAuthCredentials: CredentialsMetaResponse[];
  oAuthCallback: (
    code: string,
    state_token: string,
  ) => Promise<CredentialsMetaResponse>;
  createAPIKeyCredentials: (
    credentials: APIKeyCredentialsCreatable,
  ) => Promise<CredentialsMetaResponse>;
  deleteCredentials: (id: string) => Promise<CredentialsDeleteResponse>;
};

export type CredentialsProvidersContextType = {
  [key in CredentialsProviderName]?: CredentialsProviderData;
};

export const CredentialsProvidersContext =
  createContext<CredentialsProvidersContextType | null>(null);

export default function CredentialsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [providers, setProviders] =
    useState<CredentialsProvidersContextType | null>(null);
  const api = useMemo(() => new AutoGPTServerAPI(), []);

  const addCredentials = useCallback(
    (
      provider: CredentialsProviderName,
      credentials: CredentialsMetaResponse,
    ) => {
      setProviders((prev) => {
        if (!prev || !prev[provider]) return prev;

        const updatedProvider = { ...prev[provider] };

        if (credentials.type === "api_key") {
          updatedProvider.savedApiKeys = [
            ...updatedProvider.savedApiKeys,
            credentials,
          ];
        } else if (credentials.type === "oauth2") {
          updatedProvider.savedOAuthCredentials = [
            ...updatedProvider.savedOAuthCredentials,
            credentials,
          ];
        }

        return {
          ...prev,
          [provider]: updatedProvider,
        };
      });
    },
    [setProviders],
  );

  /** Wraps `AutoGPTServerAPI.oAuthCallback`, and adds the result to the internal credentials store. */
  const oAuthCallback = useCallback(
    async (
      provider: CredentialsProviderName,
      code: string,
      state_token: string,
    ): Promise<CredentialsMetaResponse> => {
      const credsMeta = await api.oAuthCallback(provider, code, state_token);
      addCredentials(provider, credsMeta);
      return credsMeta;
    },
    [api, addCredentials],
  );

  /** Wraps `AutoGPTServerAPI.createAPIKeyCredentials`, and adds the result to the internal credentials store. */
  const createAPIKeyCredentials = useCallback(
    async (
      provider: CredentialsProviderName,
      credentials: APIKeyCredentialsCreatable,
    ): Promise<CredentialsMetaResponse> => {
      const credsMeta = await api.createAPIKeyCredentials({
        provider,
        ...credentials,
      });
      addCredentials(provider, credsMeta);
      return credsMeta;
    },
    [api, addCredentials],
  );

  /** Wraps `AutoGPTServerAPI.deleteCredentials`, and removes the credentials from the internal store. */
  const deleteCredentials = useCallback(
    async (
      provider: CredentialsProviderName,
      id: string,
    ): Promise<CredentialsDeleteResponse> => {
      const result = await api.deleteCredentials(provider, id);
      setProviders((prev) => {
        if (!prev || !prev[provider]) return prev;

        const updatedProvider = { ...prev[provider] };
        updatedProvider.savedApiKeys = updatedProvider.savedApiKeys.filter(
          (cred) => cred.id !== id,
        );
        updatedProvider.savedOAuthCredentials =
          updatedProvider.savedOAuthCredentials.filter(
            (cred) => cred.id !== id,
          );

        return {
          ...prev,
          [provider]: updatedProvider,
        };
      });
      return result;
    },
    [api],
  );

  useEffect(() => {
    api.isAuthenticated().then((isAuthenticated) => {
      if (!isAuthenticated) return;

      CREDENTIALS_PROVIDER_NAMES.forEach(
        (provider: CredentialsProviderName) => {
          api.listCredentials(provider).then((response) => {
            const { oauthCreds, apiKeys } = response.reduce<{
              oauthCreds: CredentialsMetaResponse[];
              apiKeys: CredentialsMetaResponse[];
            }>(
              (acc, cred) => {
                if (cred.type === "oauth2") {
                  acc.oauthCreds.push(cred);
                } else if (cred.type === "api_key") {
                  acc.apiKeys.push(cred);
                }
                return acc;
              },
              { oauthCreds: [], apiKeys: [] },
            );

            setProviders((prev) => ({
              ...prev,
              [provider]: {
                provider,
                providerName: providerDisplayNames[provider],
                savedApiKeys: apiKeys,
                savedOAuthCredentials: oauthCreds,
                oAuthCallback: (code: string, state_token: string) =>
                  oAuthCallback(provider, code, state_token),
                createAPIKeyCredentials: (
                  credentials: APIKeyCredentialsCreatable,
                ) => createAPIKeyCredentials(provider, credentials),
                deleteCredentials: (id: string) =>
                  deleteCredentials(provider, id),
              },
            }));
          });
        },
      );
    });
  }, [api, createAPIKeyCredentials, deleteCredentials, oAuthCallback]);

  return (
    <CredentialsProvidersContext.Provider value={providers}>
      {children}
    </CredentialsProvidersContext.Provider>
  );
}
