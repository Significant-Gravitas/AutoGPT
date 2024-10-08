import AutoGPTServerAPI, {
  APIKeyCredentials,
  CredentialsMetaResponse,
} from "@/lib/autogpt-server-api";
import {
  createContext,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

// --8<-- [start:CredentialsProviderNames]
const CREDENTIALS_PROVIDER_NAMES = ["github", "google", "notion"] as const;

type CredentialsProviderName = (typeof CREDENTIALS_PROVIDER_NAMES)[number];

const providerDisplayNames: Record<CredentialsProviderName, string> = {
  github: "GitHub",
  google: "Google",
  notion: "Notion",
};
// --8<-- [end:CredentialsProviderNames]

type APIKeyCredentialsCreatable = Omit<
  APIKeyCredentials,
  "id" | "provider" | "type"
>;

export type CredentialsProviderData = {
  provider: string;
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

  useEffect(() => {
    api.isAuthenticated().then((isAuthenticated) => {
      if (!isAuthenticated) return;

      CREDENTIALS_PROVIDER_NAMES.forEach((provider) => {
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
            },
          }));
        });
      });
    });
  }, [api, createAPIKeyCredentials, oAuthCallback]);

  return (
    <CredentialsProvidersContext.Provider value={providers}>
      {children}
    </CredentialsProvidersContext.Provider>
  );
}
