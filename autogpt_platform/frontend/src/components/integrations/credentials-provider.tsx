import {
  APIKeyCredentials,
  CredentialsDeleteNeedConfirmationResponse,
  CredentialsDeleteResponse,
  CredentialsMetaResponse,
  CredentialsProviderName,
  PROVIDER_NAMES,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { createContext, useCallback, useEffect, useState } from "react";

// Get keys from CredentialsProviderName type
const CREDENTIALS_PROVIDER_NAMES = Object.values(
  PROVIDER_NAMES,
) as CredentialsProviderName[];

// --8<-- [start:CredentialsProviderNames]
const providerDisplayNames: Record<CredentialsProviderName, string> = {
  anthropic: "Anthropic",
  discord: "Discord",
  d_id: "D-ID",
  e2b: "E2B",
  github: "GitHub",
  google: "Google",
  google_maps: "Google Maps",
  groq: "Groq",
  ideogram: "Ideogram",
  jina: "Jina",
  medium: "Medium",
  notion: "Notion",
  nvidia: "Nvidia",
  ollama: "Ollama",
  openai: "OpenAI",
  openweathermap: "OpenWeatherMap",
  open_router: "Open Router",
  pinecone: "Pinecone",
  slant3d: "Slant3D",
  replicate: "Replicate",
  fal: "FAL",
  revid: "Rev.ID",
  twitter: "Twitter",
  unreal_speech: "Unreal Speech",
  exa: "Exa",
  hubspot: "Hubspot",
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
  deleteCredentials: (
    id: string,
    force?: boolean,
  ) => Promise<
    CredentialsDeleteResponse | CredentialsDeleteNeedConfirmationResponse
  >;
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
  const api = useBackendAPI();

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

  /** Wraps `BackendAPI.oAuthCallback`, and adds the result to the internal credentials store. */
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

  /** Wraps `BackendAPI.createAPIKeyCredentials`, and adds the result to the internal credentials store. */
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

  /** Wraps `BackendAPI.deleteCredentials`, and removes the credentials from the internal store. */
  const deleteCredentials = useCallback(
    async (
      provider: CredentialsProviderName,
      id: string,
      force: boolean = false,
    ): Promise<
      CredentialsDeleteResponse | CredentialsDeleteNeedConfirmationResponse
    > => {
      const result = await api.deleteCredentials(provider, id, force);
      if (!result.deleted) {
        return result;
      }
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

      api.listCredentials().then((response) => {
        const credentialsByProvider = response.reduce(
          (acc, cred) => {
            if (!acc[cred.provider]) {
              acc[cred.provider] = { oauthCreds: [], apiKeys: [] };
            }
            if (cred.type === "oauth2") {
              acc[cred.provider].oauthCreds.push(cred);
            } else if (cred.type === "api_key") {
              acc[cred.provider].apiKeys.push(cred);
            }
            return acc;
          },
          {} as Record<
            CredentialsProviderName,
            {
              oauthCreds: CredentialsMetaResponse[];
              apiKeys: CredentialsMetaResponse[];
            }
          >,
        );

        setProviders((prev) => ({
          ...prev,
          ...Object.fromEntries(
            CREDENTIALS_PROVIDER_NAMES.map((provider) => [
              provider,
              {
                provider,
                providerName:
                  providerDisplayNames[provider as CredentialsProviderName],
                savedApiKeys: credentialsByProvider[provider]?.apiKeys ?? [],
                savedOAuthCredentials:
                  credentialsByProvider[provider]?.oauthCreds ?? [],
                oAuthCallback: (code: string, state_token: string) =>
                  oAuthCallback(
                    provider as CredentialsProviderName,
                    code,
                    state_token,
                  ),
                createAPIKeyCredentials: (
                  credentials: APIKeyCredentialsCreatable,
                ) =>
                  createAPIKeyCredentials(
                    provider as CredentialsProviderName,
                    credentials,
                  ),
                deleteCredentials: (id: string, force: boolean = false) =>
                  deleteCredentials(
                    provider as CredentialsProviderName,
                    id,
                    force,
                  ),
              },
            ]),
          ),
        }));
      });
    });
  }, [api, createAPIKeyCredentials, deleteCredentials, oAuthCallback]);

  return (
    <CredentialsProvidersContext.Provider value={providers}>
      {children}
    </CredentialsProvidersContext.Provider>
  );
}
