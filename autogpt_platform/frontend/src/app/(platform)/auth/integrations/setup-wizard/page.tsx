"use client";

import { useGetOauthGetOauthAppInfo } from "@/app/api/__generated__/endpoints/oauth/oauth";
import { okData } from "@/app/api/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import { CredentialsInput } from "@/components/contextual/CredentialsInput/CredentialsInput";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import type {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
  CredentialsType,
} from "@/lib/autogpt-server-api";
import { CheckIcon, CircleIcon } from "@phosphor-icons/react";
import Image from "next/image";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useMemo, useRef, useState } from "react";

// All credential types - we accept any type of credential
const ALL_CREDENTIAL_TYPES: CredentialsType[] = [
  "api_key",
  "oauth2",
  "user_password",
  "host_scoped",
];

/**
 * Provider configuration for the setup wizard.
 *
 * Query parameters:
 * - `providers`: base64-encoded JSON array of { provider, scopes? } objects
 * - `app_name`: (optional) Name of the requesting application
 * - `redirect_uri`: Where to redirect after completion
 * - `state`: Anti-CSRF token
 *
 * Example `providers` JSON:
 * [
 *   { "provider": "google", "scopes": ["https://www.googleapis.com/auth/gmail.readonly"] },
 *   { "provider": "github", "scopes": ["repo"] }
 * ]
 *
 * Example URL:
 * /auth/integrations/setup-wizard?app_name=My%20App&providers=W3sicHJvdmlkZXIiOiJnb29nbGUifV0=&redirect_uri=...
 */
interface ProviderConfig {
  provider: string;
  scopes?: string[];
}

function createSchemaFromProviderConfig(
  config: ProviderConfig,
): BlockIOCredentialsSubSchema {
  return {
    type: "object",
    properties: {},
    credentials_provider: [config.provider],
    credentials_types: ALL_CREDENTIAL_TYPES,
    credentials_scopes: config.scopes,
    discriminator: undefined,
    discriminator_mapping: undefined,
    discriminator_values: undefined,
  };
}

function toDisplayName(provider: string): string {
  // Convert snake_case or kebab-case to Title Case
  return provider
    .split(/[_-]/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function parseProvidersParam(providersParam: string): ProviderConfig[] {
  try {
    // Decode base64 and parse JSON
    const decoded = atob(providersParam);
    const parsed = JSON.parse(decoded);

    if (!Array.isArray(parsed)) {
      console.warn("providers parameter is not an array");
      return [];
    }

    return parsed.filter(
      (item): item is ProviderConfig =>
        typeof item === "object" &&
        item !== null &&
        typeof item.provider === "string",
    );
  } catch (error) {
    console.warn("Failed to parse providers parameter:", error);
    return [];
  }
}

export default function IntegrationSetupWizardPage() {
  const searchParams = useSearchParams();

  // Extract query parameters
  // `providers` is a base64-encoded JSON array of { provider, scopes?: string[] } objects
  const clientID = searchParams.get("client_id");
  const providersParam = searchParams.get("providers");
  const redirectURI = searchParams.get("redirect_uri");
  const state = searchParams.get("state");

  const { data: appInfo } = useGetOauthGetOauthAppInfo(clientID || "", {
    query: { enabled: !!clientID, select: okData },
  });

  // Parse providers from base64-encoded JSON
  const providerConfigs = useMemo<ProviderConfig[]>(() => {
    if (!providersParam) return [];
    return parseProvidersParam(providersParam);
  }, [providersParam]);

  // Track selected credentials for each provider
  const [selectedCredentials, setSelectedCredentials] = useState<
    Record<string, CredentialsMetaInput | undefined>
  >({});

  // Track if we've already redirected
  const hasRedirectedRef = useRef(false);

  // Check if all providers have credentials
  const isAllComplete = useMemo(() => {
    if (providerConfigs.length === 0) return false;
    return providerConfigs.every(
      (config) => selectedCredentials[config.provider],
    );
  }, [providerConfigs, selectedCredentials]);

  // Handle credential selection
  const handleCredentialSelect = (
    provider: string,
    credential?: CredentialsMetaInput,
  ) => {
    setSelectedCredentials((prev) => ({
      ...prev,
      [provider]: credential,
    }));
  };

  // Handle completion - redirect back to client
  const handleComplete = () => {
    if (!redirectURI || hasRedirectedRef.current) return;
    hasRedirectedRef.current = true;

    const params = new URLSearchParams({
      success: "true",
    });
    if (state) {
      params.set("state", state);
    }

    window.location.href = `${redirectURI}?${params.toString()}`;
  };

  // Handle cancel - redirect back to client with error
  const handleCancel = () => {
    if (!redirectURI || hasRedirectedRef.current) return;
    hasRedirectedRef.current = true;

    const params = new URLSearchParams({
      error: "user_cancelled",
      error_description: "User cancelled the integration setup",
    });
    if (state) {
      params.set("state", state);
    }

    window.location.href = `${redirectURI}?${params.toString()}`;
  };

  // Validate required parameters
  const missingParams: string[] = [];
  if (!providersParam) missingParams.push("providers");
  if (!redirectURI) missingParams.push("redirect_uri");

  if (missingParams.length > 0) {
    return (
      <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
        <AuthCard title="Invalid Request">
          <ErrorCard
            context="request details"
            responseError={{
              message: `Missing required parameters: ${missingParams.join(", ")}`,
            }}
            hint="Please contact the administrator of the app that sent you here."
            isOurProblem={false}
          />
        </AuthCard>
      </div>
    );
  }

  if (providerConfigs.length === 0) {
    return (
      <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
        <AuthCard title="Invalid Request">
          <ErrorCard
            context="providers"
            responseError={{ message: "No providers specified" }}
            hint="Please contact the administrator of the app that sent you here."
            isOurProblem={false}
          />
          <Button
            variant="secondary"
            onClick={handleCancel}
            className="mt-4 w-full"
          >
            Cancel
          </Button>
        </AuthCard>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
      <AuthCard title="Connect Your Accounts">
        <div className="flex w-full flex-col gap-6">
          <Text variant="body" className="text-center text-slate-600">
            {appInfo ? (
              <>
                <strong>{appInfo.name}</strong> is requesting you to connect the
                following integrations to your AutoGPT account.
              </>
            ) : (
              "Please connect the following integrations to continue."
            )}
          </Text>

          {/* Provider credentials list */}
          <div className="space-y-4">
            {providerConfigs.map((config) => {
              const schema = createSchemaFromProviderConfig(config);
              const isSelected = !!selectedCredentials[config.provider];

              return (
                <div
                  key={config.provider}
                  className="relative rounded-xl border border-slate-200 bg-white p-4"
                >
                  <div className="mb-4 flex items-center gap-2">
                    <div className="relative size-8">
                      <Image
                        src={`/integrations/${config.provider}.png`}
                        alt={`${config.provider} icon`}
                        fill
                        className="object-contain group-disabled:opacity-50"
                      />
                    </div>
                    <Text className="mx-1" variant="large-medium">
                      {toDisplayName(config.provider)}
                    </Text>
                    <div className="grow"></div>
                    {isSelected ? (
                      <CheckIcon
                        size={20}
                        className="text-green-500"
                        weight="bold"
                      />
                    ) : (
                      <CircleIcon
                        size={20}
                        className="text-slate-300"
                        weight="bold"
                      />
                    )}
                    {isSelected && (
                      <Text variant="small" className="text-green-600">
                        Connected
                      </Text>
                    )}
                  </div>

                  <CredentialsInput
                    schema={schema}
                    selectedCredentials={selectedCredentials[config.provider]}
                    onSelectCredentials={(credMeta) =>
                      handleCredentialSelect(config.provider, credMeta)
                    }
                    showTitle={false}
                    className="mb-0"
                  />
                </div>
              );
            })}
          </div>

          {/* Action buttons */}
          <div className="flex flex-col gap-3">
            <Button
              variant="primary"
              onClick={handleComplete}
              disabled={!isAllComplete}
              className="w-full text-lg"
            >
              {isAllComplete
                ? "Continue"
                : `Connect ${providerConfigs.length - Object.values(selectedCredentials).filter(Boolean).length} more`}
            </Button>
            <Button
              variant="secondary"
              onClick={handleCancel}
              className="w-full text-lg"
            >
              Cancel
            </Button>
          </div>

          {/* Link to integrations settings */}
          <Text variant="small" className="text-center text-slate-500">
            You can view and manage all your integrations in your{" "}
            <Link
              href="/profile/integrations"
              target="_blank"
              className="text-purple-600 underline hover:text-purple-800"
            >
              integration settings
            </Link>
            .
          </Text>
        </div>
      </AuthCard>
    </div>
  );
}
