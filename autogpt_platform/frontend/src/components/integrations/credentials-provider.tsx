import { createContext, useCallback, useEffect, useState } from "react";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import {
  APIKeyCredentials,
  CredentialsDeleteNeedConfirmationResponse,
  CredentialsDeleteResponse,
  CredentialsMetaResponse,
  CredentialsProviderName,
  HostScopedCredentials,
  PROVIDER_NAMES,
  UserPasswordCredentials,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useToastOnFail } from "@/components/ui/use-toast";

// Get keys from CredentialsProviderName type
const CREDENTIALS_PROVIDER_NAMES = Object.values(
  PROVIDER_NAMES,
) as CredentialsProviderName[];

// --8<-- [start:CredentialsProviderNames]
const providerDisplayNames: Record<CredentialsProviderName, string> = {
  aiml_api: "AI/ML",
  anthropic: "Anthropic",
  apollo: "Apollo",
  discord: "Discord",
  d_id: "D-ID",
  e2b: "E2B",
  exa: "Exa",
  fal: "FAL",
  github: "GitHub",
  google: "Google",
  google_maps: "Google Maps",
  groq: "Groq",
  http: "HTTP",
  hubspot: "Hubspot",
  ideogram: "Ideogram",
  jina: "Jina",
  linear: "Linear",
  medium: "Medium",
  mem0: "Mem0",
  notion: "Notion",
  nvidia: "Nvidia",
  ollama: "Ollama",
  openai: "OpenAI",
  openweathermap: "OpenWeatherMap",
  open_router: "Open Router",
  llama_api: "Llama API",
  pinecone: "Pinecone",
  screenshotone: "ScreenshotOne",
  slant3d: "Slant3D",
  smartlead: "SmartLead",
  smtp: "SMTP",
  reddit: "Reddit",
  replicate: "Replicate",
  revid: "Rev.ID",
  twitter: "Twitter",
  todoist: "Todoist",
  unreal_speech: "Unreal Speech",
  zerobounce: "ZeroBounce",
} as const;
// --8<-- [end:CredentialsProviderNames]

type APIKeyCredentialsCreatable = Omit<
  APIKeyCredentials,
  "id" | "provider" | "type"
>;

type UserPasswordCredentialsCreatable = Omit<
  UserPasswordCredentials,
  "id" | "provider" | "type"
>;

type HostScopedCredentialsCreatable = Omit<
  HostScopedCredentials,
  "id" | "provider" | "type"
>;

export type CredentialsProviderData = {
  provider: CredentialsProviderName;
  providerName: string;
  savedCredentials: CredentialsMetaResponse[];
  oAuthCallback: (
    code: string,
    state_token: string,
  ) => Promise<CredentialsMetaResponse>;
  createAPIKeyCredentials: (
    credentials: APIKeyCredentialsCreatable,
  ) => Promise<CredentialsMetaResponse>;
  createUserPasswordCredentials: (
    credentials: UserPasswordCredentialsCreatable,
  ) => Promise<CredentialsMetaResponse>;
  createHostScopedCredentials: (
    credentials: HostScopedCredentialsCreatable,
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
  const { isLoggedIn } = useSupabase();
  const api = useBackendAPI();
  const onFailToast = useToastOnFail();

  const addCredentials = useCallback(
    (
      provider: CredentialsProviderName,
      credentials: CredentialsMetaResponse,
    ) => {
      setProviders((prev) => {
        if (!prev || !prev[provider]) return prev;

        return {
          ...prev,
          [provider]: {
            ...prev[provider],
            savedCredentials: [...prev[provider].savedCredentials, credentials],
          },
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
      try {
        const credsMeta = await api.oAuthCallback(provider, code, state_token);
        addCredentials(provider, credsMeta);
        return credsMeta;
      } catch (error) {
        onFailToast("complete OAuth authentication")(error);
        throw error;
      }
    },
    [api, addCredentials, onFailToast],
  );

  /** Wraps `BackendAPI.createAPIKeyCredentials`, and adds the result to the internal credentials store. */
  const createAPIKeyCredentials = useCallback(
    async (
      provider: CredentialsProviderName,
      credentials: APIKeyCredentialsCreatable,
    ): Promise<CredentialsMetaResponse> => {
      try {
        const credsMeta = await api.createAPIKeyCredentials({
          provider,
          ...credentials,
        });
        addCredentials(provider, credsMeta);
        return credsMeta;
      } catch (error) {
        onFailToast("create API key credentials")(error);
        throw error;
      }
    },
    [api, addCredentials, onFailToast],
  );

  /** Wraps `BackendAPI.createUserPasswordCredentials`, and adds the result to the internal credentials store. */
  const createUserPasswordCredentials = useCallback(
    async (
      provider: CredentialsProviderName,
      credentials: UserPasswordCredentialsCreatable,
    ): Promise<CredentialsMetaResponse> => {
      try {
        const credsMeta = await api.createUserPasswordCredentials({
          provider,
          ...credentials,
        });
        addCredentials(provider, credsMeta);
        return credsMeta;
      } catch (error) {
        onFailToast("create user/password credentials")(error);
        throw error;
      }
    },
    [api, addCredentials, onFailToast],
  );

  /** Wraps `BackendAPI.createHostScopedCredentials`, and adds the result to the internal credentials store. */
  const createHostScopedCredentials = useCallback(
    async (
      provider: CredentialsProviderName,
      credentials: HostScopedCredentialsCreatable,
    ): Promise<CredentialsMetaResponse> => {
      try {
        const credsMeta = await api.createHostScopedCredentials({
          provider,
          ...credentials,
        });
        addCredentials(provider, credsMeta);
        return credsMeta;
      } catch (error) {
        onFailToast("create host-scoped credentials")(error);
        throw error;
      }
    },
    [api, addCredentials, onFailToast],
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
      try {
        const result = await api.deleteCredentials(provider, id, force);
        if (!result.deleted) {
          return result;
        }
        setProviders((prev) => {
          if (!prev || !prev[provider]) return prev;

          return {
            ...prev,
            [provider]: {
              ...prev[provider],
              savedCredentials: prev[provider].savedCredentials.filter(
                (cred) => cred.id !== id,
              ),
            },
          };
        });
        return result;
      } catch (error) {
        onFailToast("delete credentials")(error);
        throw error;
      }
    },
    [api, onFailToast],
  );

  useEffect(() => {
    if (!isLoggedIn) {
      if (isLoggedIn == false) setProviders(null);
      return;
    }

    api
      .listCredentials()
      .then((response) => {
        const credentialsByProvider = response.reduce(
          (acc, cred) => {
            if (!acc[cred.provider]) {
              acc[cred.provider] = [];
            }
            acc[cred.provider].push(cred);
            return acc;
          },
          {} as Record<CredentialsProviderName, CredentialsMetaResponse[]>,
        );

        setProviders((prev) => ({
          ...prev,
          ...Object.fromEntries(
            CREDENTIALS_PROVIDER_NAMES.map((provider) => [
              provider,
              {
                provider,
                providerName: providerDisplayNames[provider],
                savedCredentials: credentialsByProvider[provider] ?? [],
                oAuthCallback: (code: string, state_token: string) =>
                  oAuthCallback(provider, code, state_token),
                createAPIKeyCredentials: (
                  credentials: APIKeyCredentialsCreatable,
                ) => createAPIKeyCredentials(provider, credentials),
                createUserPasswordCredentials: (
                  credentials: UserPasswordCredentialsCreatable,
                ) => createUserPasswordCredentials(provider, credentials),
                createHostScopedCredentials: (
                  credentials: HostScopedCredentialsCreatable,
                ) => createHostScopedCredentials(provider, credentials),
                deleteCredentials: (id: string, force: boolean = false) =>
                  deleteCredentials(provider, id, force),
              } satisfies CredentialsProviderData,
            ]),
          ),
        }));
      })
      .catch(onFailToast("load credentials"));
  }, [
    api,
    isLoggedIn,
    createAPIKeyCredentials,
    createUserPasswordCredentials,
    createHostScopedCredentials,
    deleteCredentials,
    oAuthCallback,
  ]);

  return (
    <CredentialsProvidersContext.Provider value={providers}>
      {children}
    </CredentialsProvidersContext.Provider>
  );
}
