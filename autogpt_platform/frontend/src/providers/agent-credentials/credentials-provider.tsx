import { useToastOnFail } from "@/components/molecules/Toast/use-toast";
import {
  APIKeyCredentials,
  CredentialsDeleteNeedConfirmationResponse,
  CredentialsDeleteResponse,
  CredentialsMetaResponse,
  CredentialsProviderName,
  HostScopedCredentials,
  UserPasswordCredentials,
} from "@/lib/autogpt-server-api";
import { postV2ExchangeOauthCodeForMcpTokens } from "@/app/api/__generated__/endpoints/mcp/mcp";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { toDisplayName } from "@/providers/agent-credentials/helper";
import { createContext, useCallback, useEffect, useState } from "react";

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
  /** Whether this provider has platform credits available (system credentials) */
  isSystemProvider: boolean;
  oAuthCallback: (
    code: string,
    state_token: string,
  ) => Promise<CredentialsMetaResponse>;
  /** MCP-specific OAuth callback that uses dynamic per-server OAuth discovery. */
  mcpOAuthCallback: (
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
  const [providerNames, setProviderNames] = useState<string[]>([]);
  const [systemProviders, setSystemProviders] = useState<Set<string>>(
    new Set(),
  );
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
        const credsMeta = await api.oAuthCallback(
          provider as string,
          code,
          state_token,
        );
        addCredentials(provider as string, credsMeta);
        return credsMeta;
      } catch (error) {
        onFailToast("complete OAuth authentication")(error);
        throw error;
      }
    },
    [api, addCredentials, onFailToast],
  );

  /** Exchanges an MCP OAuth code for tokens and adds the result to the internal credentials store. */
  const mcpOAuthCallback = useCallback(
    async (
      code: string,
      state_token: string,
    ): Promise<CredentialsMetaResponse> => {
      try {
        const response = await postV2ExchangeOauthCodeForMcpTokens({
          code,
          state_token,
        });
        if (response.status !== 200) throw response.data;
        const credsMeta: CredentialsMetaResponse = {
          ...response.data,
          title: response.data.title ?? undefined,
          scopes: response.data.scopes ?? undefined,
          username: response.data.username ?? undefined,
          host: response.data.host ?? undefined,
        };
        addCredentials("mcp", credsMeta);
        return credsMeta;
      } catch (error) {
        onFailToast("complete MCP OAuth authentication")(error);
        throw error;
      }
    },
    [addCredentials, onFailToast],
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
        const result = await api.deleteCredentials(
          provider as string,
          id,
          force,
        );
        if (!result.deleted) {
          return result;
        }
        setProviders((prev) => {
          if (!prev || !prev[provider]) return prev;

          return {
            ...prev,
            [provider]: {
              ...prev[provider]!,
              savedCredentials: prev[provider]!.savedCredentials.filter(
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

  const loadCredentials = useCallback(() => {
    if (!isLoggedIn || providerNames.length === 0) {
      if (isLoggedIn == false) setProviders({});
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
            providerNames.map((provider) => {
              const providerCredentials = credentialsByProvider[provider] ?? [];

              return [
                provider,
                {
                  provider,
                  providerName: toDisplayName(provider as string),
                  savedCredentials: providerCredentials,
                  isSystemProvider: systemProviders.has(provider),
                  oAuthCallback: (code: string, state_token: string) =>
                    oAuthCallback(provider, code, state_token),
                  mcpOAuthCallback,
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
              ];
            }),
          ),
        }));
      })
      .catch(onFailToast("load credentials"));
  }, [
    api,
    isLoggedIn,
    providerNames,
    systemProviders,
    createAPIKeyCredentials,
    createUserPasswordCredentials,
    createHostScopedCredentials,
    deleteCredentials,
    oAuthCallback,
    mcpOAuthCallback,
    onFailToast,
  ]);

  // Fetch provider names and system providers on mount
  useEffect(() => {
    Promise.all([api.listProviders(), api.listSystemProviders()])
      .then(([names, systemList]) => {
        setProviderNames(names);
        setSystemProviders(new Set(systemList));
      })
      .catch(onFailToast("Load provider names"));
  }, [api, onFailToast]);

  useEffect(() => {
    loadCredentials();
  }, [loadCredentials]);

  return (
    <CredentialsProvidersContext.Provider value={providers}>
      {children}
    </CredentialsProvidersContext.Provider>
  );
}
