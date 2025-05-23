import {
  APIKeyCredentials,
  CredentialsDeleteNeedConfirmationResponse,
  CredentialsDeleteResponse,
  CredentialsMetaResponse,
  UserPasswordCredentials,
} from "@/lib/autogpt-server-api";
import {
  CredentialsProviderName,
  providerDisplayNames,
  PROVIDER_NAMES,
} from "@/lib/provider-meta";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { createContext, useCallback, useEffect, useState } from "react";

// Get keys from CredentialsProviderName type
const CREDENTIALS_PROVIDER_NAMES = Object.values(
  PROVIDER_NAMES,
) as CredentialsProviderName[];

// --8<-- [start:CredentialsProviderNames]
// Re-exported from the shared provider metadata
// --8<-- [end:CredentialsProviderNames]

type APIKeyCredentialsCreatable = Omit<
  APIKeyCredentials,
  "id" | "provider" | "type"
>;

type UserPasswordCredentialsCreatable = Omit<
  UserPasswordCredentials,
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

  /** Wraps `BackendAPI.createUserPasswordCredentials`, and adds the result to the internal credentials store. */
  const createUserPasswordCredentials = useCallback(
    async (
      provider: CredentialsProviderName,
      credentials: UserPasswordCredentialsCreatable,
    ): Promise<CredentialsMetaResponse> => {
      const credsMeta = await api.createUserPasswordCredentials({
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
                deleteCredentials: (id: string, force: boolean = false) =>
                  deleteCredentials(provider, id, force),
              } satisfies CredentialsProviderData,
            ]),
          ),
        }));
      });
    });
  }, [
    api,
    createAPIKeyCredentials,
    createUserPasswordCredentials,
    deleteCredentials,
    oAuthCallback,
  ]);

  return (
    <CredentialsProvidersContext.Provider value={providers}>
      {children}
    </CredentialsProvidersContext.Provider>
  );
}
