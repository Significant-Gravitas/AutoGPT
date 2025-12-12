"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { getWebSocketToken } from "@/lib/supabase/actions";

// Module-level flag to prevent duplicate requests across React StrictMode re-renders
// This is keyed by session_id to allow different sessions
const attemptedSessions = new Set<string>();

interface ScopeInfo {
  scope: string;
  description: string;
}

interface ClientInfo {
  name: string;
  logo_url: string | null;
  privacy_policy_url: string | null;
  terms_url: string | null;
}

interface ConsentData {
  needs_consent: true;
  consent_token: string;
  client: ClientInfo;
  scopes: ScopeInfo[];
  action_url: string;
}

interface RedirectData {
  redirect_url: string;
  needs_consent: false;
}

interface ErrorData {
  error: string;
  error_description: string;
  redirect_url?: string;
}

type ResumeResponse = ConsentData | RedirectData | ErrorData;

function isConsentData(data: ResumeResponse): data is ConsentData {
  return "needs_consent" in data && data.needs_consent === true;
}

function isRedirectData(data: ResumeResponse): data is RedirectData {
  return "redirect_url" in data && !("error" in data);
}

function isErrorData(data: ResumeResponse): data is ErrorData {
  return "error" in data;
}

/**
 * OAuth Consent Form Component
 *
 * Renders a proper React component for the consent form instead of dangerouslySetInnerHTML
 */
function ConsentForm({
  client,
  scopes,
  consentToken,
  actionUrl,
}: {
  client: ClientInfo;
  scopes: ScopeInfo[];
  consentToken: string;
  actionUrl: string;
}) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const backendUrl = process.env.NEXT_PUBLIC_AGPT_SERVER_URL;
  const backendOrigin = backendUrl
    ? new URL(backendUrl).origin
    : "http://localhost:8006";

  // Full action URL for form submission
  const fullActionUrl = `${backendOrigin}${actionUrl}`;

  function handleSubmit() {
    setIsSubmitting(true);
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-900 to-slate-800 p-5">
      <div className="w-full max-w-md rounded-2xl bg-zinc-800 p-8 shadow-2xl">
        {/* Header */}
        <div className="mb-6 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-xl bg-zinc-700">
            {client.logo_url ? (
              <img
                src={client.logo_url}
                alt={client.name}
                className="h-12 w-12 rounded-lg"
              />
            ) : (
              <span className="text-3xl text-zinc-400">
                {client.name.charAt(0).toUpperCase()}
              </span>
            )}
          </div>
          <h1 className="text-xl font-semibold text-zinc-100">
            Authorize <span className="text-cyan-400">{client.name}</span>
          </h1>
          <p className="mt-2 text-sm text-zinc-400">
            wants to access your AutoGPT account
          </p>
        </div>

        {/* Divider */}
        <div className="my-6 h-px bg-zinc-700" />

        {/* Scopes Section */}
        <div className="mb-6">
          <h2 className="mb-4 text-sm font-medium text-zinc-400">
            This will allow {client.name} to:
          </h2>
          <div className="space-y-3">
            {scopes.map((scope) => (
              <div
                key={scope.scope}
                className="flex items-start gap-3 border-b border-zinc-700 pb-3 last:border-0"
              >
                <svg
                  className="mt-0.5 h-5 w-5 flex-shrink-0 text-cyan-400"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
                <span className="text-sm leading-relaxed text-zinc-300">
                  {scope.description}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Form */}
        <form method="POST" action={fullActionUrl} onSubmit={handleSubmit}>
          <input type="hidden" name="consent_token" value={consentToken} />
          <div className="flex gap-3">
            <button
              type="submit"
              name="authorize"
              value="false"
              disabled={isSubmitting}
              className="flex-1 rounded-lg bg-zinc-700 px-6 py-3 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              name="authorize"
              value="true"
              disabled={isSubmitting}
              className="flex-1 rounded-lg bg-cyan-400 px-6 py-3 text-sm font-medium text-slate-900 transition-colors hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isSubmitting ? "Authorizing..." : "Allow"}
            </button>
          </div>
        </form>

        {/* Footer Links */}
        {(client.privacy_policy_url || client.terms_url) && (
          <div className="mt-6 text-center text-xs text-zinc-500">
            {client.privacy_policy_url && (
              <a
                href={client.privacy_policy_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-zinc-400 hover:underline"
              >
                Privacy Policy
              </a>
            )}
            {client.privacy_policy_url && client.terms_url && (
              <span className="mx-2">•</span>
            )}
            {client.terms_url && (
              <a
                href={client.terms_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-zinc-400 hover:underline"
              >
                Terms of Service
              </a>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * OAuth Resume Page
 *
 * This page handles resuming the OAuth authorization flow after a user logs in.
 * It fetches the consent data from the backend via JSON API and renders the consent form.
 */
export default function OAuthResumePage() {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session_id");
  const { isUserLoading, refreshSession } = useSupabase();

  const [consentData, setConsentData] = useState<ConsentData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const retryCountRef = useRef(0);
  const maxRetries = 5;

  const resumeOAuthFlow = useCallback(async () => {
    // Prevent multiple attempts for the same session (handles React StrictMode)
    if (!sessionId) {
      setError(
        "Missing session ID. Please start the authorization process again.",
      );
      setIsLoading(false);
      return;
    }

    if (attemptedSessions.has(sessionId)) {
      // Already attempted this session, don't retry
      return;
    }

    if (isUserLoading) {
      return; // Wait for auth state to load
    }

    // Mark this session as attempted IMMEDIATELY to prevent duplicate requests
    attemptedSessions.add(sessionId);

    try {
      // Get the access token from server action (which reads cookies properly)
      let tokenResult = await getWebSocketToken();
      let accessToken = tokenResult.token;

      // If no token, retry a few times with delays
      while (!accessToken && retryCountRef.current < maxRetries) {
        retryCountRef.current += 1;
        console.log(
          `Retrying to get access token (attempt ${retryCountRef.current}/${maxRetries})...`,
        );

        // Try refreshing the session
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

      // Call the backend resume endpoint with JSON accept header
      const backendUrl = process.env.NEXT_PUBLIC_AGPT_SERVER_URL;
      if (!backendUrl) {
        setError("Backend URL not configured.");
        setIsLoading(false);
        return;
      }

      // Extract the origin from the backend URL
      let backendOrigin: string;
      try {
        const url = new URL(backendUrl);
        backendOrigin = url.origin;
      } catch {
        setError("Invalid backend URL configuration.");
        setIsLoading(false);
        return;
      }

      // Use Accept: application/json to get JSON response instead of HTML
      // This solves the CORS/redirect issue by letting us handle redirects client-side
      const response = await fetch(
        `${backendOrigin}/oauth/authorize/resume?session_id=${encodeURIComponent(sessionId)}`,
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
          setError(
            `Authorization failed (${response.status}). Please try again.`,
          );
        }
        setIsLoading(false);
        return;
      }

      // Handle redirect response (user already authorized these scopes)
      if (isRedirectData(data)) {
        window.location.href = data.redirect_url;
        return;
      }

      // Handle consent required
      if (isConsentData(data)) {
        setConsentData(data);
        setIsLoading(false);
        return;
      }

      // Unexpected response
      setError("Unexpected response from server. Please try again.");
      setIsLoading(false);
    } catch (err) {
      console.error("OAuth resume error:", err);
      setError(
        "An error occurred while resuming authorization. Please try again.",
      );
      setIsLoading(false);
    }
  }, [sessionId, isUserLoading, refreshSession]);

  useEffect(() => {
    resumeOAuthFlow();
  }, [resumeOAuthFlow]);

  if (isLoading || isUserLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="text-center">
          <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-zinc-600 border-t-cyan-400"></div>
          <p className="text-zinc-400">Resuming authorization...</p>
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
            Authorization Error
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

  if (consentData) {
    return (
      <ConsentForm
        client={consentData.client}
        scopes={consentData.scopes}
        consentToken={consentData.consent_token}
        actionUrl={consentData.action_url}
      />
    );
  }

  return null;
}
