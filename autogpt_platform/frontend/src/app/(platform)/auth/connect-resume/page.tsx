"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { getWebSocketToken } from "@/lib/supabase/actions";

// Module-level flag to prevent duplicate requests across React StrictMode re-renders
const attemptedSessions = new Set<string>();

interface ScopeInfo {
  scope: string;
  description: string;
}

interface CredentialInfo {
  id: string;
  title: string;
  username: string;
}

interface ClientInfo {
  name: string;
  logo_url: string | null;
}

interface ConnectData {
  connect_token: string;
  client: ClientInfo;
  provider: string;
  scopes: ScopeInfo[];
  credentials: CredentialInfo[];
  action_url: string;
}

interface ErrorData {
  error: string;
  error_description: string;
}

type ResumeResponse = ConnectData | ErrorData;

function isConnectData(data: ResumeResponse): data is ConnectData {
  return "connect_token" in data;
}

function isErrorData(data: ResumeResponse): data is ErrorData {
  return "error" in data;
}

/**
 * Connect Consent Form Component
 *
 * Renders a proper React component for the integration connect consent form
 */
function ConnectForm({
  client,
  provider,
  scopes,
  credentials,
  connectToken,
  actionUrl,
}: {
  client: ClientInfo;
  provider: string;
  scopes: ScopeInfo[];
  credentials: CredentialInfo[];
  connectToken: string;
  actionUrl: string;
}) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedCredential, setSelectedCredential] = useState<string>(
    credentials.length > 0 ? credentials[0].id : "",
  );

  const backendUrl = process.env.NEXT_PUBLIC_AGPT_SERVER_URL;
  const backendOrigin = backendUrl
    ? new URL(backendUrl).origin
    : "http://localhost:8006";

  const fullActionUrl = `${backendOrigin}${actionUrl}`;

  function handleSubmit() {
    setIsSubmitting(true);
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-900 to-slate-800 p-5">
      <div className="w-full max-w-md rounded-2xl bg-zinc-800 p-8 shadow-2xl">
        {/* Header */}
        <div className="mb-6 text-center">
          <h1 className="text-xl font-semibold text-zinc-100">
            Connect{" "}
            <span className="rounded bg-zinc-700 px-2 py-1 text-sm capitalize">
              {provider}
            </span>
          </h1>
          <p className="mt-2 text-sm text-zinc-400">
            <span className="font-semibold text-cyan-400">{client.name}</span>{" "}
            wants to use your {provider} integration
          </p>
        </div>

        {/* Divider */}
        <div className="my-6 h-px bg-zinc-700" />

        {/* Scopes Section */}
        <div className="mb-6">
          <h2 className="mb-4 text-sm font-medium text-zinc-400">
            This will allow {client.name} to:
          </h2>
          <div className="space-y-2">
            {scopes.map((scope) => (
              <div key={scope.scope} className="flex items-start gap-2 py-2">
                <span className="flex-shrink-0 text-cyan-400">&#10003;</span>
                <span className="text-sm text-zinc-300">
                  {scope.description}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Divider */}
        <div className="my-6 h-px bg-zinc-700" />

        {/* Form */}
        <form method="POST" action={fullActionUrl} onSubmit={handleSubmit}>
          <input type="hidden" name="connect_token" value={connectToken} />

          {/* Existing credentials selection */}
          {credentials.length > 0 && (
            <>
              <h3 className="mb-3 text-sm font-medium text-zinc-400">
                Select an existing credential:
              </h3>
              <div className="mb-4 space-y-2">
                {credentials.map((cred) => (
                  <label
                    key={cred.id}
                    className={`flex cursor-pointer items-center gap-3 rounded-lg border p-3 transition-colors ${
                      selectedCredential === cred.id
                        ? "border-cyan-400 bg-cyan-400/10"
                        : "border-zinc-700 hover:border-cyan-400/50"
                    }`}
                  >
                    <input
                      type="radio"
                      name="credential_id"
                      value={cred.id}
                      checked={selectedCredential === cred.id}
                      onChange={() => setSelectedCredential(cred.id)}
                      className="hidden"
                    />
                    <div>
                      <div className="text-sm font-medium text-zinc-200">
                        {cred.title}
                      </div>
                      {cred.username && (
                        <div className="text-xs text-zinc-500">
                          {cred.username}
                        </div>
                      )}
                    </div>
                  </label>
                ))}
              </div>
              <div className="my-4 h-px bg-zinc-700" />
            </>
          )}

          {/* Connect new account */}
          <div className="mb-4">
            {credentials.length > 0 ? (
              <h3 className="mb-3 text-sm font-medium text-zinc-400">
                Or connect a new account:
              </h3>
            ) : (
              <p className="mb-3 text-sm text-zinc-400">
                You don&apos;t have any {provider} credentials yet.
              </p>
            )}
            <button
              type="submit"
              name="action"
              value="connect_new"
              disabled={isSubmitting}
              className="w-full rounded-lg bg-blue-500 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-blue-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Connect {provider.charAt(0).toUpperCase() + provider.slice(1)}{" "}
              Account
            </button>
          </div>

          {/* Action buttons */}
          <div className="flex gap-3">
            <button
              type="submit"
              name="action"
              value="deny"
              disabled={isSubmitting}
              className="flex-1 rounded-lg bg-zinc-700 px-6 py-3 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Cancel
            </button>
            {credentials.length > 0 && (
              <button
                type="submit"
                name="action"
                value="approve"
                disabled={isSubmitting}
                className="flex-1 rounded-lg bg-cyan-400 px-6 py-3 text-sm font-medium text-slate-900 transition-colors hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isSubmitting ? "Approving..." : "Approve"}
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

/**
 * Connect Resume Page
 *
 * This page handles resuming the integration connect flow after a user logs in.
 * It fetches the connect data from the backend via JSON API and renders the consent form.
 */
export default function ConnectResumePage() {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session_id");
  const { isUserLoading, refreshSession } = useSupabase();

  const [connectData, setConnectData] = useState<ConnectData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const retryCountRef = useRef(0);
  const maxRetries = 5;

  const resumeConnectFlow = useCallback(async () => {
    if (!sessionId) {
      setError(
        "Missing session ID. Please start the connection process again.",
      );
      setIsLoading(false);
      return;
    }

    if (attemptedSessions.has(sessionId)) {
      return;
    }

    if (isUserLoading) {
      return;
    }

    attemptedSessions.add(sessionId);

    try {
      let tokenResult = await getWebSocketToken();
      let accessToken = tokenResult.token;

      while (!accessToken && retryCountRef.current < maxRetries) {
        retryCountRef.current += 1;
        console.log(
          `Retrying to get access token (attempt ${retryCountRef.current}/${maxRetries})...`,
        );
        await refreshSession();
        await new Promise((resolve) => setTimeout(resolve, 1000));
        tokenResult = await getWebSocketToken();
        accessToken = tokenResult.token;
      }

      if (!accessToken) {
        setError(
          "Unable to retrieve authentication token. Please log in again.",
        );
        setIsLoading(false);
        return;
      }

      const backendUrl = process.env.NEXT_PUBLIC_AGPT_SERVER_URL;
      if (!backendUrl) {
        setError("Backend URL not configured.");
        setIsLoading(false);
        return;
      }

      let backendOrigin: string;
      try {
        const url = new URL(backendUrl);
        backendOrigin = url.origin;
      } catch {
        setError("Invalid backend URL configuration.");
        setIsLoading(false);
        return;
      }

      const response = await fetch(
        `${backendOrigin}/connect/resume?session_id=${encodeURIComponent(sessionId)}`,
        {
          method: "GET",
          headers: {
            Authorization: `Bearer ${accessToken}`,
            Accept: "application/json",
          },
        },
      );

      const data: ResumeResponse = await response.json();

      if (!response.ok) {
        if (isErrorData(data)) {
          setError(data.error_description || data.error);
        } else {
          setError(`Connection failed (${response.status}). Please try again.`);
        }
        setIsLoading(false);
        return;
      }

      if (isConnectData(data)) {
        setConnectData(data);
        setIsLoading(false);
        return;
      }

      setError("Unexpected response from server. Please try again.");
      setIsLoading(false);
    } catch (err) {
      console.error("Connect resume error:", err);
      setError(
        "An error occurred while resuming connection. Please try again.",
      );
      setIsLoading(false);
    }
  }, [sessionId, isUserLoading, refreshSession]);

  useEffect(() => {
    resumeConnectFlow();
  }, [resumeConnectFlow]);

  if (isLoading || isUserLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="text-center">
          <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-zinc-600 border-t-cyan-400"></div>
          <p className="text-zinc-400">Resuming connection...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="mx-auto max-w-md rounded-2xl bg-zinc-800 p-8 text-center shadow-2xl">
          <div className="mx-auto mb-4 h-16 w-16 text-red-500">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
          </div>
          <h1 className="mb-2 text-xl font-semibold text-red-400">
            Connection Error
          </h1>
          <p className="mb-6 text-zinc-400">{error}</p>
          <button
            onClick={() => window.close()}
            className="rounded-lg bg-zinc-700 px-6 py-3 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-600"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  if (connectData) {
    return (
      <ConnectForm
        client={connectData.client}
        provider={connectData.provider}
        scopes={connectData.scopes}
        credentials={connectData.credentials}
        connectToken={connectData.connect_token}
        actionUrl={connectData.action_url}
      />
    );
  }

  return null;
}
