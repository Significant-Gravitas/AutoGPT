import AutoGPTServerAPI, { CredentialsMetaResponse } from "@/lib/autogpt-server-api";
import { useRouter } from "next/navigation";
import { createContext, useCallback, useEffect, useMemo, useState } from "react";

const CredentialsProvidersNames = ["github", "google", "notion"] as const;

type CredentialsProviderType = typeof CredentialsProvidersNames[number];

export type CredentialsProviderData = {
  provider: string;
  providerName: string;
  savedApiKeys: CredentialsMetaResponse[];
  savedOAuthCredentials: CredentialsMetaResponse[];
  oAuthLogin: (scopes?: string[]) => Promise<void>;
}

export type CredentialsProvidersContextType = {
  [key in CredentialsProviderType]?: CredentialsProviderData;
}

export const CredentialsProvidersContext = createContext<CredentialsProvidersContextType | null>(null);

export default function CredentialsProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [providers, setProviders] = useState<CredentialsProvidersContextType | null>(null);
  const api = useMemo(() => new AutoGPTServerAPI(), []);

  const providerName: Record<CredentialsProviderType, string> = {
    github: "GitHub",
    google: "Google",
    notion: "Notion",
  };

  useEffect(() => {

    CredentialsProvidersNames.forEach((provider) => {
      api
        .listCredentials(provider)
        .then((response) => {
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
              providerName: providerName[provider],
              savedApiKeys: apiKeys,
              savedOAuthCredentials: oauthCreds,
              oAuthLogin: (scopes?: string[]) => oAuthLogin(provider, scopes),
            },
          }));
        });
    });
  }, [api]);

  const oAuthLogin = useCallback(
    async (provider: CredentialsProviderType, scopes?: string[]) => {
      const { login_url } = await api.oAuthLogin(provider, scopes);
      router.push(login_url);
    },
    [api],
  );

  return (
    <CredentialsProvidersContext.Provider value={providers}>
      {children}
    </CredentialsProvidersContext.Provider>
  );
}
