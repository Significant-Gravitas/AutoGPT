/**
 * Environment configuration helper
 *
 * Provides unified access to environment variables with server-side priority.
 * Server-side code uses Docker service names, client-side falls back to localhost.
 */

import { isServerSide } from "./utils/is-server-side";

export interface VercelConfig {
  isVercelEnvExposed: boolean;
  isVercelCI: boolean;
  vercelEnvironment: "production" | "preview" | "development" | null;
  vercelTargetEnv: "production" | "staging" | "development" | string | null;
  // The value represents the domain name of the deployment (e.g., *.vercel.app) and does not include the protocol scheme (https://).
  vercelUrl: string | null;
  vercelBranchUrl: string | null;
  vercelProjectProductionUrl: string | null;
  vercelRegion: string | null;
  vercelDeploymentId: string | null;
  vercelProjectId: string | null;
  vercelGitRepoSlug: string | null;
  vercelGitRepoOwner: string | null;
  vercelGitCommitRef: string | null;
  vercelGitCommitSha: string | null;
}

/**
 * Gets the AGPT server URL with server-side priority
 * Server-side: Uses AGPT_SERVER_URL (http://rest_server:8006/api)
 * Client-side: Falls back to NEXT_PUBLIC_AGPT_SERVER_URL (http://localhost:8006/api)
 */
export function getAgptServerApiUrl(): string {
  // If server-side and server URL exists, use it
  if (isServerSide() && process.env.AGPT_SERVER_URL) {
    return process.env.AGPT_SERVER_URL;
  }

  // Otherwise use the public URL
  return process.env.NEXT_PUBLIC_AGPT_SERVER_URL || "http://localhost:8006/api";
}

export function getAgptServerBaseUrl(): string {
  return getAgptServerApiUrl().replace("/api", "");
}

export function getVercelEnv(): VercelConfig {
  const toNullable = (v?: string) => (v && v.length > 0 ? v : null);

  // Check if running on Vercel by checking the VERCEL environment variable
  const isVercelEnvExposed = process.env.VERCEL === "1";
  const isVercelCI = process.env.CI === "1";

  const ve = process.env.VERCEL_ENV;
  const vercelEnvironment: VercelConfig["vercelEnvironment"] =
    ve === "production" || ve === "preview" || ve === "development" ? ve : null;

  return {
    isVercelEnvExposed,
    isVercelCI,
    vercelEnvironment,
    vercelTargetEnv: toNullable(process.env.VERCEL_TARGET_ENV),
    vercelUrl: toNullable(process.env.VERCEL_URL),
    vercelBranchUrl: toNullable(process.env.VERCEL_BRANCH_URL),
    vercelProjectProductionUrl: toNullable(
      process.env.VERCEL_PROJECT_PRODUCTION_URL,
    ),
    vercelRegion: toNullable(process.env.VERCEL_REGION),
    vercelDeploymentId: toNullable(process.env.VERCEL_DEPLOYMENT_ID),
    vercelProjectId: toNullable(process.env.VERCEL_PROJECT_ID),
    vercelGitRepoSlug: toNullable(process.env.VERCEL_GIT_REPO_SLUG),
    vercelGitRepoOwner: toNullable(process.env.VERCEL_GIT_REPO_OWNER),
    vercelGitCommitRef: toNullable(process.env.VERCEL_GIT_COMMIT_REF),
    vercelGitCommitSha: toNullable(process.env.VERCEL_GIT_COMMIT_SHA),
  };
}

/**
 * Gets the AGPT WebSocket URL with server-side priority
 * Server-side: Uses AGPT_WS_SERVER_URL (ws://websocket_server:8001/ws)
 * Client-side: Falls back to NEXT_PUBLIC_AGPT_WS_SERVER_URL (ws://localhost:8001/ws)
 */
export function getAgptWsServerUrl(): string {
  // If server-side and server URL exists, use it
  if (isServerSide() && process.env.AGPT_WS_SERVER_URL) {
    return process.env.AGPT_WS_SERVER_URL;
  }

  // Otherwise use the public URL
  return process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL || "ws://localhost:8001/ws";
}

/**
 * Gets the Supabase URL with server-side priority
 * Server-side: Uses SUPABASE_URL (http://kong:8000)
 * Client-side: Falls back to NEXT_PUBLIC_SUPABASE_URL (http://localhost:8000)
 */
export function getSupabaseUrl(): string {
  // If server-side and server URL exists, use it
  if (isServerSide() && process.env.SUPABASE_URL) {
    return process.env.SUPABASE_URL;
  }

  // Otherwise use the public URL
  return process.env.NEXT_PUBLIC_SUPABASE_URL || "http://localhost:8000";
}

/**
 * Gets the Supabase anon key
 * Uses NEXT_PUBLIC_SUPABASE_ANON_KEY since anon keys are public and same across environments
 */
export function getSupabaseAnonKey(): string {
  return process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";
}
