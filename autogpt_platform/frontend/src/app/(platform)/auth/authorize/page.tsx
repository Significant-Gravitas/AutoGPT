"use client";

import { useState } from "react";
import { useSearchParams } from "next/navigation";
import { AuthCard } from "@/components/auth/AuthCard";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ImageIcon, SealCheckIcon } from "@phosphor-icons/react";
import {
  postOauthAuthorize,
  useGetOauthGetOauthAppInfo,
} from "@/app/api/__generated__/endpoints/oauth/oauth";
import type { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";

// Human-readable scope descriptions
const SCOPE_DESCRIPTIONS: { [key in APIKeyPermission]: string } = {
  IDENTITY: "View your user ID, e-mail, and timezone",
  EXECUTE_GRAPH: "Run your agents",
  READ_GRAPH: "View your agents and their configurations",
  WRITE_GRAPH: "Create agent graphs",
  WRITE_LIBRARY: "Add agents to your library",
  EXECUTE_BLOCK: "Execute individual blocks",
  READ_BLOCK: "View available blocks",
  READ_STORE: "Access the Marketplace",
  USE_TOOLS: "Use tools on your behalf",
  MANAGE_INTEGRATIONS: "Set up new integrations",
  READ_INTEGRATIONS: "View your connected integrations",
  DELETE_INTEGRATIONS: "Remove connected integrations",
};

export default function AuthorizePage() {
  const searchParams = useSearchParams();

  // Extract OAuth parameters from URL
  const clientID = searchParams.get("client_id");
  const redirectURI = searchParams.get("redirect_uri");
  const scope = searchParams.get("scope");
  const state = searchParams.get("state");
  const codeChallenge = searchParams.get("code_challenge");
  const codeChallengeMethod =
    searchParams.get("code_challenge_method") || "S256";
  const responseType = searchParams.get("response_type") || "code";

  // Parse requested scopes
  const requestedScopes = scope?.split(" ").filter(Boolean) || [];

  // Fetch application info using generated hook
  const {
    data: appInfoResponse,
    isLoading,
    error,
    refetch,
  } = useGetOauthGetOauthAppInfo(clientID || "", {
    query: {
      enabled: !!clientID,
      staleTime: Infinity,
      refetchOnMount: false,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
    },
  });

  const appInfo = appInfoResponse?.status === 200 ? appInfoResponse.data : null;

  // Validate required parameters
  const missingParams: string[] = [];
  if (!clientID) missingParams.push("client_id");
  if (!redirectURI) missingParams.push("redirect_uri");
  if (!scope) missingParams.push("scope");
  if (!state) missingParams.push("state");
  if (!codeChallenge) missingParams.push("code_challenge");

  const [isAuthorizing, setIsAuthorizing] = useState(false);
  const [authorizeError, setAuthorizeError] = useState<string | null>(null);

  async function handleApprove() {
    setIsAuthorizing(true);
    setAuthorizeError(null);

    try {
      // Call the backend /oauth/authorize POST endpoint
      // Returns JSON with redirect_url that we use to redirect the user
      const response = await postOauthAuthorize({
        client_id: clientID!,
        redirect_uri: redirectURI!,
        scopes: requestedScopes,
        state: state!,
        response_type: responseType,
        code_challenge: codeChallenge!,
        code_challenge_method: codeChallengeMethod as "S256" | "plain",
      });

      if (response.status === 200 && response.data.redirect_url) {
        window.location.href = response.data.redirect_url;
      } else {
        setAuthorizeError("Authorization failed: no redirect URL received");
        setIsAuthorizing(false);
      }
    } catch (err) {
      console.error("Authorization error:", err);
      setAuthorizeError(
        err instanceof Error ? err.message : "Authorization failed",
      );
      setIsAuthorizing(false);
    }
  }

  function handleDeny() {
    // Redirect back to client with access_denied error
    const params = new URLSearchParams({
      error: "access_denied",
      error_description: "User denied access",
      state: state || "",
    });
    window.location.href = `${redirectURI}?${params.toString()}`;
  }

  // Show error if missing required parameters
  if (missingParams.length > 0) {
    return (
      <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
        <AuthCard title="Invalid Request">
          <ErrorCard
            context="request parameters"
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

  // Show loading state
  if (isLoading) {
    return (
      <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
        <AuthCard title="Loading...">
          <div className="flex flex-col items-center gap-4 py-8">
            <LoadingSpinner size="large" />
            <Text variant="body" className="text-center text-slate-500">
              Loading application information...
            </Text>
          </div>
        </AuthCard>
      </div>
    );
  }

  // Show error if app not found
  if (error || !appInfo) {
    return (
      <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
        <AuthCard title="Application Not Found">
          <ErrorCard
            context="application"
            responseError={
              error
                ? error
                : {
                    message:
                      "The application you're trying to authorize could not be found or is disabled.",
                  }
            }
            onRetry={refetch}
          />
          {redirectURI && (
            <Button
              variant="secondary"
              onClick={handleDeny}
              className="mt-4 w-full"
            >
              Return to Application
            </Button>
          )}
        </AuthCard>
      </div>
    );
  }

  // Validate that requested scopes are allowed by the app
  const invalidScopes = requestedScopes.filter(
    (s) => !appInfo.scopes.includes(s),
  );

  if (invalidScopes.length > 0) {
    return (
      <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
        <AuthCard title="Invalid Scopes">
          <ErrorCard
            context="scopes"
            responseError={{
              message: `The application is requesting scopes it is not authorized for: ${invalidScopes.join(", ")}`,
            }}
            hint="Please contact the administrator of the app that sent you here."
            isOurProblem={false}
          />
          <Button
            variant="secondary"
            onClick={handleDeny}
            className="mt-4 w-full"
          >
            Return to Application
          </Button>
        </AuthCard>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
      <AuthCard title="Authorize Application">
        <div className="flex w-full flex-col gap-6">
          {/* App info */}
          <div className="flex flex-col items-center text-center">
            {/* App logo */}
            <div className="mb-4 flex size-16 items-center justify-center overflow-hidden rounded-xl border bg-slate-100">
              {appInfo.logo_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={appInfo.logo_url}
                  alt={`${appInfo.name} logo`}
                  className="h-full w-full object-cover"
                />
              ) : (
                <ImageIcon className="h-8 w-8 text-slate-400" />
              )}
            </div>
            <Text variant="h4" className="mb-2">
              {appInfo.name}
            </Text>
            {appInfo.description && (
              <Text variant="body" className="text-slate-600">
                {appInfo.description}
              </Text>
            )}
          </div>

          {/* Permissions */}
          <div>
            <Text variant="body-medium" className="mb-3">
              This application is requesting permission to:
            </Text>
            <ul className="space-y-2">
              {requestedScopes.map((scopeKey) => (
                <li key={scopeKey} className="flex items-start gap-3">
                  <SealCheckIcon className="mt-0.5 text-green-600" />
                  <Text variant="body">
                    {SCOPE_DESCRIPTIONS[scopeKey as APIKeyPermission] ||
                      scopeKey}
                  </Text>
                </li>
              ))}
            </ul>
          </div>

          {/* Error message */}
          {authorizeError && (
            <ErrorCard
              context="authorization"
              responseError={{ message: authorizeError }}
            />
          )}

          {/* Action buttons */}
          <div className="flex flex-col gap-3">
            <Button
              variant="primary"
              onClick={handleApprove}
              disabled={isAuthorizing}
              className="w-full text-lg"
            >
              {isAuthorizing ? "Authorizing..." : "Authorize"}
            </Button>
            <Button
              variant="secondary"
              onClick={handleDeny}
              disabled={isAuthorizing}
              className="w-full text-lg"
            >
              Deny
            </Button>
          </div>

          {/* Warning */}
          <Text variant="small" className="text-center text-slate-500">
            By authorizing, you allow this application to access your AutoGPT
            account with the permissions listed above.
          </Text>
        </div>
      </AuthCard>
    </div>
  );
}
