# Next.js Integration Guide

This guide shows how to integrate AutoGPT's OAuth popup flow into a Next.js application, enabling your users to connect their credentials and execute agents.

## Prerequisites

- A Next.js 13+ application (App Router recommended)
- An OAuth client registered with AutoGPT (see [External API Integration Guide](../external-api-integration.md))
- Your `client_id` and `client_secret`

## Installation

No additional packages are required. The integration uses standard browser APIs and Next.js features.

## Project Structure

```
src/
├── app/
│   ├── api/
│   │   └── autogpt/
│   │       ├── callback/route.ts    # OAuth callback handler
│   │       └── webhook/route.ts     # Webhook handler
│   └── connect/
│       └── page.tsx                 # Connect button page
├── lib/
│   └── autogpt/
│       ├── client.ts               # AutoGPT API client
│       ├── oauth.ts                # OAuth utilities
│       └── types.ts                # TypeScript types
└── components/
    └── ConnectButton.tsx           # Reusable connect button
```

## Step 1: Define Types

Create `src/lib/autogpt/types.ts`:

```typescript
export interface OAuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  expires_at: number; // Unix timestamp
}

export interface ConnectResult {
  type: "autogpt_connect_result";
  success: boolean;
  nonce: string;
  grant_id?: string;
  credential_id?: string;
  provider?: string;
  error?: string;
  error_description?: string;
}

export interface ExecutionResult {
  execution_id: string;
  status: "queued" | "running" | "completed" | "failed";
  started_at?: string;
  completed_at?: string;
  outputs?: Record<string, unknown>;
  error?: string;
}

export type IntegrationScope =
  | "google:gmail.readonly"
  | "google:gmail.send"
  | "google:sheets.read"
  | "google:sheets.write"
  | "google:calendar.read"
  | "google:calendar.write"
  | "google:drive.read"
  | "google:drive.write"
  | "github:repo.read"
  | "github:repo.write"
  | "github:user.read"
  | "twitter:tweet.read"
  | "twitter:tweet.write"
  | "notion:read"
  | "notion:write"
  | "slack:read"
  | "slack:write";
```

## Step 2: Create OAuth Utilities

Create `src/lib/autogpt/oauth.ts`:

```typescript
const AUTOGPT_BASE_URL =
  process.env.NEXT_PUBLIC_AUTOGPT_URL || "https://platform.agpt.co";

/**
 * Generate a cryptographically secure code verifier for PKCE
 */
export function generateCodeVerifier(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return base64UrlEncode(array);
}

/**
 * Generate code challenge from verifier using SHA-256
 */
export async function generateCodeChallenge(verifier: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return base64UrlEncode(new Uint8Array(digest));
}

/**
 * Base64 URL encode a buffer
 */
function base64UrlEncode(buffer: Uint8Array): string {
  return btoa(String.fromCharCode(...buffer))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

/**
 * Build the OAuth authorization URL
 */
export async function buildAuthorizationUrl(
  clientId: string,
  redirectUri: string,
  scopes: string[]
): Promise<{ url: string; state: string; codeVerifier: string }> {
  const state = crypto.randomUUID();
  const codeVerifier = generateCodeVerifier();
  const codeChallenge = await generateCodeChallenge(codeVerifier);

  const url = new URL(`${AUTOGPT_BASE_URL}/oauth/authorize`);
  url.searchParams.set("response_type", "code");
  url.searchParams.set("client_id", clientId);
  url.searchParams.set("redirect_uri", redirectUri);
  url.searchParams.set("state", state);
  url.searchParams.set("code_challenge", codeChallenge);
  url.searchParams.set("code_challenge_method", "S256");
  url.searchParams.set("scope", scopes.join(" "));

  return { url: url.toString(), state, codeVerifier };
}

/**
 * Exchange authorization code for tokens
 */
export async function exchangeCodeForTokens(
  code: string,
  codeVerifier: string,
  clientId: string,
  clientSecret: string,
  redirectUri: string
): Promise<{
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}> {
  const response = await fetch(`${AUTOGPT_BASE_URL}/oauth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      code,
      redirect_uri: redirectUri,
      client_id: clientId,
      client_secret: clientSecret,
      code_verifier: codeVerifier,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error_description || error.error || "Token exchange failed");
  }

  return response.json();
}

/**
 * Refresh an access token
 */
export async function refreshAccessToken(
  refreshToken: string,
  clientId: string,
  clientSecret: string
): Promise<{
  access_token: string;
  refresh_token: string;
  expires_in: number;
}> {
  const response = await fetch(`${AUTOGPT_BASE_URL}/oauth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: refreshToken,
      client_id: clientId,
      client_secret: clientSecret,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error_description || error.error || "Token refresh failed");
  }

  return response.json();
}
```

## Step 3: Create the AutoGPT Client

Create `src/lib/autogpt/client.ts`:

```typescript
import type {
  OAuthTokens,
  ConnectResult,
  ExecutionResult,
  IntegrationScope,
} from "./types";

const AUTOGPT_BASE_URL =
  process.env.NEXT_PUBLIC_AUTOGPT_URL || "https://platform.agpt.co";
const CLIENT_ID = process.env.NEXT_PUBLIC_AUTOGPT_CLIENT_ID!;

export class AutoGPTClient {
  private tokens: OAuthTokens;
  private onTokenRefresh?: (tokens: OAuthTokens) => void;

  constructor(tokens: OAuthTokens, onTokenRefresh?: (tokens: OAuthTokens) => void) {
    this.tokens = tokens;
    this.onTokenRefresh = onTokenRefresh;
  }

  /**
   * Get the current access token, refreshing if necessary
   */
  private async getAccessToken(): Promise<string> {
    // Check if token is expired (with 60s buffer)
    if (this.tokens.expires_at < Date.now() / 1000 + 60) {
      await this.refreshTokens();
    }
    return this.tokens.access_token;
  }

  /**
   * Refresh the OAuth tokens
   */
  private async refreshTokens(): Promise<void> {
    const response = await fetch("/api/autogpt/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: this.tokens.refresh_token }),
    });

    if (!response.ok) {
      throw new Error("Failed to refresh tokens");
    }

    const newTokens = await response.json();
    this.tokens = {
      ...newTokens,
      expires_at: Math.floor(Date.now() / 1000) + newTokens.expires_in,
    };

    this.onTokenRefresh?.(this.tokens);
  }

  /**
   * Open the Connect popup to request credential grants
   */
  requestGrant(
    provider: string,
    scopes: IntegrationScope[]
  ): Promise<ConnectResult> {
    return new Promise((resolve, reject) => {
      const nonce = crypto.randomUUID();

      const url = new URL(`${AUTOGPT_BASE_URL}/connect/${provider}`);
      url.searchParams.set("client_id", CLIENT_ID);
      url.searchParams.set("scopes", scopes.join(","));
      url.searchParams.set("nonce", nonce);
      url.searchParams.set("redirect_origin", window.location.origin);

      // Calculate popup position (centered)
      const width = 500;
      const height = 600;
      const left = window.screenX + (window.outerWidth - width) / 2;
      const top = window.screenY + (window.outerHeight - height) / 2;

      const popup = window.open(
        url.toString(),
        "AutoGPT Connect",
        `width=${width},height=${height},left=${left},top=${top},popup=true`
      );

      if (!popup) {
        reject(new Error("Failed to open popup. Please allow popups for this site."));
        return;
      }

      // Poll to check if popup was closed without completing
      const pollTimer = setInterval(() => {
        if (popup.closed) {
          clearInterval(pollTimer);
          window.removeEventListener("message", handler);
          reject(new Error("Popup was closed"));
        }
      }, 500);

      const handler = (event: MessageEvent) => {
        // Verify origin
        if (event.origin !== AUTOGPT_BASE_URL) return;

        const data = event.data as ConnectResult;
        if (data?.type !== "autogpt_connect_result") return;
        if (data?.nonce !== nonce) return;

        clearInterval(pollTimer);
        window.removeEventListener("message", handler);
        popup.close();

        if (data.success) {
          resolve(data);
        } else {
          reject(new Error(data.error_description || data.error || "Connection failed"));
        }
      };

      window.addEventListener("message", handler);
    });
  }

  /**
   * Execute an agent
   */
  async executeAgent(
    agentId: string,
    inputs: Record<string, unknown>,
    options?: {
      grantIds?: string[];
      webhookUrl?: string;
    }
  ): Promise<ExecutionResult> {
    const accessToken = await this.getAccessToken();

    const response = await fetch(
      `${AUTOGPT_BASE_URL}/api/external/v1/executions/agents/${agentId}/execute`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          inputs,
          grant_ids: options?.grantIds,
          webhook_url: options?.webhookUrl,
        }),
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Execution failed");
    }

    return response.json();
  }

  /**
   * Get execution status
   */
  async getExecutionStatus(executionId: string): Promise<ExecutionResult> {
    const accessToken = await this.getAccessToken();

    const response = await fetch(
      `${AUTOGPT_BASE_URL}/api/external/v1/executions/${executionId}`,
      {
        headers: { Authorization: `Bearer ${accessToken}` },
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to get execution status");
    }

    return response.json();
  }

  /**
   * Wait for execution to complete with polling
   */
  async waitForCompletion(
    executionId: string,
    options?: {
      timeoutMs?: number;
      pollIntervalMs?: number;
      onStatusChange?: (status: ExecutionResult) => void;
    }
  ): Promise<ExecutionResult> {
    const timeoutMs = options?.timeoutMs ?? 300000; // 5 minutes default
    const pollIntervalMs = options?.pollIntervalMs ?? 2000;
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const status = await this.getExecutionStatus(executionId);
      options?.onStatusChange?.(status);

      if (status.status === "completed") {
        return status;
      }

      if (status.status === "failed") {
        throw new Error(status.error || "Execution failed");
      }

      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    }

    throw new Error("Execution timeout");
  }
}
```

## Step 4: Create the OAuth Callback Route

Create `src/app/api/autogpt/callback/route.ts`:

```typescript
import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { exchangeCodeForTokens } from "@/lib/autogpt/oauth";

const CLIENT_ID = process.env.AUTOGPT_CLIENT_ID!;
const CLIENT_SECRET = process.env.AUTOGPT_CLIENT_SECRET!;
const REDIRECT_URI = process.env.AUTOGPT_REDIRECT_URI!;

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const code = searchParams.get("code");
  const state = searchParams.get("state");
  const error = searchParams.get("error");

  // Handle OAuth errors
  if (error) {
    const errorDescription = searchParams.get("error_description") || error;
    return NextResponse.redirect(
      new URL(`/connect?error=${encodeURIComponent(errorDescription)}`, request.url)
    );
  }

  if (!code || !state) {
    return NextResponse.redirect(
      new URL("/connect?error=Missing+code+or+state", request.url)
    );
  }

  // Verify state
  const cookieStore = await cookies();
  const storedState = cookieStore.get("autogpt_oauth_state")?.value;
  const codeVerifier = cookieStore.get("autogpt_code_verifier")?.value;

  if (state !== storedState || !codeVerifier) {
    return NextResponse.redirect(
      new URL("/connect?error=Invalid+state", request.url)
    );
  }

  try {
    // Exchange code for tokens
    const tokens = await exchangeCodeForTokens(
      code,
      codeVerifier,
      CLIENT_ID,
      CLIENT_SECRET,
      REDIRECT_URI
    );

    // Store tokens securely (use your preferred method)
    // Option 1: HTTP-only cookie (shown here)
    // Option 2: Server-side session
    // Option 3: Encrypted in database

    const response = NextResponse.redirect(new URL("/connect?success=true", request.url));

    // Store tokens in HTTP-only cookie
    response.cookies.set("autogpt_tokens", JSON.stringify({
      ...tokens,
      expires_at: Math.floor(Date.now() / 1000) + tokens.expires_in,
    }), {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 30, // 30 days
    });

    // Clear OAuth state cookies
    response.cookies.delete("autogpt_oauth_state");
    response.cookies.delete("autogpt_code_verifier");

    return response;
  } catch (error) {
    console.error("OAuth callback error:", error);
    return NextResponse.redirect(
      new URL(`/connect?error=${encodeURIComponent(String(error))}`, request.url)
    );
  }
}
```

## Step 5: Create the Token Refresh Route

Create `src/app/api/autogpt/refresh/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";
import { refreshAccessToken } from "@/lib/autogpt/oauth";

const CLIENT_ID = process.env.AUTOGPT_CLIENT_ID!;
const CLIENT_SECRET = process.env.AUTOGPT_CLIENT_SECRET!;

export async function POST(request: NextRequest) {
  try {
    const { refresh_token } = await request.json();

    if (!refresh_token) {
      return NextResponse.json(
        { error: "Missing refresh token" },
        { status: 400 }
      );
    }

    const tokens = await refreshAccessToken(
      refresh_token,
      CLIENT_ID,
      CLIENT_SECRET
    );

    return NextResponse.json(tokens);
  } catch (error) {
    console.error("Token refresh error:", error);
    return NextResponse.json(
      { error: "Failed to refresh token" },
      { status: 401 }
    );
  }
}
```

## Step 6: Create the Webhook Handler

Create `src/app/api/autogpt/webhook/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";

const WEBHOOK_SECRET = process.env.AUTOGPT_WEBHOOK_SECRET;

interface WebhookPayload {
  event: "execution.started" | "execution.completed" | "execution.failed" | "grant.revoked";
  timestamp: string;
  data: {
    execution_id?: string;
    status?: string;
    outputs?: Record<string, unknown>;
    error?: string;
    grant_id?: string;
  };
}

export async function POST(request: NextRequest) {
  const body = await request.text();

  // Verify webhook signature if secret is configured
  if (WEBHOOK_SECRET) {
    const signature = request.headers.get("x-webhook-signature");
    const timestamp = request.headers.get("x-webhook-timestamp");

    if (!signature || !timestamp) {
      return NextResponse.json({ error: "Missing signature" }, { status: 401 });
    }

    const expectedSignature = `sha256=${crypto
      .createHmac("sha256", WEBHOOK_SECRET)
      .update(body)
      .digest("hex")}`;

    if (signature !== expectedSignature) {
      return NextResponse.json({ error: "Invalid signature" }, { status: 401 });
    }

    // Check timestamp to prevent replay attacks (allow 5 minute window)
    const timestampDate = new Date(timestamp);
    const now = new Date();
    if (Math.abs(now.getTime() - timestampDate.getTime()) > 5 * 60 * 1000) {
      return NextResponse.json({ error: "Timestamp too old" }, { status: 401 });
    }
  }

  const payload: WebhookPayload = JSON.parse(body);

  // Handle webhook events
  switch (payload.event) {
    case "execution.started":
      console.log(`Execution ${payload.data.execution_id} started`);
      // Notify user, update database, etc.
      break;

    case "execution.completed":
      console.log(`Execution ${payload.data.execution_id} completed`);
      console.log("Outputs:", payload.data.outputs);
      // Store results, notify user, trigger follow-up actions
      break;

    case "execution.failed":
      console.error(`Execution ${payload.data.execution_id} failed:`, payload.data.error);
      // Handle failure, notify user, retry logic
      break;

    case "grant.revoked":
      console.log(`Grant ${payload.data.grant_id} was revoked`);
      // Update UI, disable features that depend on this grant
      break;
  }

  return NextResponse.json({ received: true });
}
```

## Step 7: Create the Connect Button Component

Create `src/components/ConnectButton.tsx`:

```typescript
"use client";

import { useState } from "react";
import { AutoGPTClient } from "@/lib/autogpt/client";
import type { OAuthTokens, IntegrationScope, ConnectResult } from "@/lib/autogpt/types";

interface ConnectButtonProps {
  tokens: OAuthTokens;
  provider: string;
  scopes: IntegrationScope[];
  onSuccess?: (result: ConnectResult) => void;
  onError?: (error: Error) => void;
  children?: React.ReactNode;
}

export function ConnectButton({
  tokens,
  provider,
  scopes,
  onSuccess,
  onError,
  children,
}: ConnectButtonProps) {
  const [isConnecting, setIsConnecting] = useState(false);

  async function handleConnect() {
    setIsConnecting(true);

    try {
      const client = new AutoGPTClient(tokens);
      const result = await client.requestGrant(provider, scopes);
      onSuccess?.(result);
    } catch (error) {
      onError?.(error as Error);
    } finally {
      setIsConnecting(false);
    }
  }

  return (
    <button
      onClick={handleConnect}
      disabled={isConnecting}
      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
    >
      {isConnecting ? "Connecting..." : children || `Connect ${provider}`}
    </button>
  );
}
```

## Step 8: Create the Connect Page

Create `src/app/connect/page.tsx`:

```typescript
"use client";

import { useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { ConnectButton } from "@/components/ConnectButton";
import { buildAuthorizationUrl } from "@/lib/autogpt/oauth";
import type { OAuthTokens, ConnectResult } from "@/lib/autogpt/types";

const CLIENT_ID = process.env.NEXT_PUBLIC_AUTOGPT_CLIENT_ID!;
const REDIRECT_URI = process.env.NEXT_PUBLIC_AUTOGPT_REDIRECT_URI!;

export default function ConnectPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [tokens, setTokens] = useState<OAuthTokens | null>(null);
  const [grants, setGrants] = useState<ConnectResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check for errors from OAuth callback
    const errorParam = searchParams.get("error");
    if (errorParam) {
      setError(errorParam);
    }

    // Load tokens from cookie (in production, fetch from server)
    async function loadTokens() {
      const response = await fetch("/api/autogpt/tokens");
      if (response.ok) {
        const data = await response.json();
        setTokens(data);
      }
    }
    loadTokens();
  }, [searchParams]);

  async function handleLogin() {
    const scopes = [
      "openid",
      "profile",
      "email",
      "agents:execute",
      "integrations:connect",
      "integrations:list",
    ];

    const { url, state, codeVerifier } = await buildAuthorizationUrl(
      CLIENT_ID,
      REDIRECT_URI,
      scopes
    );

    // Store state and verifier in cookies for validation
    document.cookie = `autogpt_oauth_state=${state}; path=/; max-age=600; SameSite=Lax`;
    document.cookie = `autogpt_code_verifier=${codeVerifier}; path=/; max-age=600; SameSite=Lax`;

    // Redirect to AutoGPT authorization
    window.location.href = url;
  }

  function handleGrantSuccess(result: ConnectResult) {
    setGrants([...grants, result]);
    setError(null);
  }

  function handleGrantError(err: Error) {
    setError(err.message);
  }

  if (!tokens) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Connect to AutoGPT</h1>
          <p className="text-gray-600 mb-6">
            Sign in with AutoGPT to use AI agents with your connected services.
          </p>
          <button
            onClick={handleLogin}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Sign in with AutoGPT
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-8">
      <h1 className="text-2xl font-bold mb-6">Connect Your Services</h1>

      {error && (
        <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-lg">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Google Connection */}
        <div className="p-6 border rounded-lg">
          <h2 className="text-xl font-semibold mb-2">Google</h2>
          <p className="text-gray-600 mb-4">
            Connect Gmail, Sheets, Calendar, and Drive
          </p>
          <ConnectButton
            tokens={tokens}
            provider="google"
            scopes={["google:gmail.readonly", "google:sheets.read"]}
            onSuccess={handleGrantSuccess}
            onError={handleGrantError}
          >
            Connect Google
          </ConnectButton>
        </div>

        {/* GitHub Connection */}
        <div className="p-6 border rounded-lg">
          <h2 className="text-xl font-semibold mb-2">GitHub</h2>
          <p className="text-gray-600 mb-4">
            Access repositories and issues
          </p>
          <ConnectButton
            tokens={tokens}
            provider="github"
            scopes={["github:repo.read", "github:user.read"]}
            onSuccess={handleGrantSuccess}
            onError={handleGrantError}
          >
            Connect GitHub
          </ConnectButton>
        </div>

        {/* Notion Connection */}
        <div className="p-6 border rounded-lg">
          <h2 className="text-xl font-semibold mb-2">Notion</h2>
          <p className="text-gray-600 mb-4">
            Read and write Notion pages
          </p>
          <ConnectButton
            tokens={tokens}
            provider="notion"
            scopes={["notion:read", "notion:write"]}
            onSuccess={handleGrantSuccess}
            onError={handleGrantError}
          >
            Connect Notion
          </ConnectButton>
        </div>
      </div>

      {/* Active Grants */}
      {grants.length > 0 && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4">Connected Services</h2>
          <ul className="space-y-2">
            {grants.map((grant) => (
              <li
                key={grant.grant_id}
                className="p-4 bg-green-50 border border-green-200 rounded-lg"
              >
                <span className="font-medium">{grant.provider}</span>
                <span className="text-gray-500 ml-2">Grant ID: {grant.grant_id}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

## Step 9: Environment Variables

Add to your `.env.local`:

```bash
# Public (exposed to browser)
NEXT_PUBLIC_AUTOGPT_URL=https://platform.agpt.co
NEXT_PUBLIC_AUTOGPT_CLIENT_ID=your_client_id_here
NEXT_PUBLIC_AUTOGPT_REDIRECT_URI=http://localhost:3000/api/autogpt/callback

# Private (server-side only)
AUTOGPT_CLIENT_ID=your_client_id_here
AUTOGPT_CLIENT_SECRET=your_client_secret_here
AUTOGPT_REDIRECT_URI=http://localhost:3000/api/autogpt/callback
AUTOGPT_WEBHOOK_SECRET=your_webhook_secret_here
```

## Complete Usage Example

Here's how to use all the pieces together:

```typescript
"use client";

import { useState } from "react";
import { AutoGPTClient } from "@/lib/autogpt/client";
import { ConnectButton } from "@/components/ConnectButton";
import type { OAuthTokens, ConnectResult, ExecutionResult } from "@/lib/autogpt/types";

export default function AgentRunner({ tokens }: { tokens: OAuthTokens }) {
  const [grants, setGrants] = useState<string[]>([]);
  const [execution, setExecution] = useState<ExecutionResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const client = new AutoGPTClient(tokens, (newTokens) => {
    // Handle token refresh - save to your storage
    console.log("Tokens refreshed");
  });

  function handleGrantSuccess(result: ConnectResult) {
    if (result.grant_id) {
      setGrants([...grants, result.grant_id]);
    }
  }

  async function runAgent() {
    setIsRunning(true);

    try {
      // Start execution
      const exec = await client.executeAgent(
        "your-agent-id",
        { query: "Search my emails for invoices from last month" },
        { grantIds: grants }
      );

      setExecution(exec);

      // Wait for completion with status updates
      const result = await client.waitForCompletion(exec.execution_id, {
        onStatusChange: (status) => setExecution(status),
      });

      console.log("Agent completed:", result.outputs);
    } catch (error) {
      console.error("Agent execution failed:", error);
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-6">Run Email Agent</h1>

      {/* Connect Google for Gmail access */}
      {grants.length === 0 && (
        <div className="mb-6">
          <p className="mb-2">First, connect your Google account:</p>
          <ConnectButton
            tokens={tokens}
            provider="google"
            scopes={["google:gmail.readonly"]}
            onSuccess={handleGrantSuccess}
            onError={(err) => console.error(err)}
          >
            Connect Gmail
          </ConnectButton>
        </div>
      )}

      {/* Run Agent */}
      {grants.length > 0 && (
        <div className="mb-6">
          <button
            onClick={runAgent}
            disabled={isRunning}
            className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
          >
            {isRunning ? "Running..." : "Run Email Search Agent"}
          </button>
        </div>
      )}

      {/* Execution Status */}
      {execution && (
        <div className="mt-6 p-4 bg-gray-100 rounded-lg">
          <h2 className="font-semibold mb-2">Execution Status</h2>
          <p>ID: {execution.execution_id}</p>
          <p>Status: {execution.status}</p>
          {execution.outputs && (
            <pre className="mt-2 p-2 bg-white rounded overflow-auto">
              {JSON.stringify(execution.outputs, null, 2)}
            </pre>
          )}
          {execution.error && (
            <p className="text-red-600 mt-2">{execution.error}</p>
          )}
        </div>
      )}
    </div>
  );
}
```

## Security Best Practices

1. **Store tokens securely** - Use HTTP-only cookies or server-side sessions
2. **Validate the state parameter** - Prevents CSRF attacks
3. **Use PKCE** - Required for all authorization flows
4. **Verify popup origin** - Only accept messages from `platform.agpt.co`
5. **Verify webhook signatures** - Prevents spoofed webhook calls
6. **Keep secrets server-side** - Never expose `client_secret` to the browser
7. **Implement token refresh** - Handle expired tokens gracefully

## Next Steps

- [External API Integration Guide](../external-api-integration.md) - Full API reference
- [Ruby on Rails Integration](./rails.md) - Server-side integration example
- [Discord Community](https://discord.gg/autogpt) - Get help from the community
