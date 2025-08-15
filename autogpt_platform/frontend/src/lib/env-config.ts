/**
 * Environment configuration helper
 *
 * Provides unified access to environment variables with server-side priority.
 * Server-side code uses Docker service names, client-side falls back to localhost.
 */

import { isServerSide } from "./utils/is-server-side";

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
