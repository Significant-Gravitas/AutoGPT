import { createContext, useCallback, useEffect, useState } from "react";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import {
  APIKeyCredentials,
  CredentialsDeleteNeedConfirmationResponse,
  CredentialsDeleteResponse,
  CredentialsMetaResponse,
  CredentialsProviderName,
  HostScopedCredentials,
  UserPasswordCredentials,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useToastOnFail } from "@/components/molecules/Toast/use-toast";
import { toDisplayName } from "@/providers/agent-credentials/helper";

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
  const [providerNames, setProviderNames] = useState<string[]>([]);
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

  // Fetch provider names on mount
  useEffect(() => {
    api
      .listProviders()
      .then((names) => {
        setProviderNames(names);
      })
      .catch(onFailToast("load provider names"));
  }, [api, onFailToast]);

  useEffect(() => {
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
            providerNames.map((provider) => [
              provider,
              {
                provider,
                providerName: toDisplayName(provider as string),
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
    providerNames,
    createAPIKeyCredentials,
    createUserPasswordCredentials,
    createHostScopedCredentials,
    deleteCredentials,
    oAuthCallback,
    onFailToast,
  ]);

  return (
    <CredentialsProvidersContext.Provider value={providers}>
      {children}
    </CredentialsProvidersContext.Provider>
  );
}
