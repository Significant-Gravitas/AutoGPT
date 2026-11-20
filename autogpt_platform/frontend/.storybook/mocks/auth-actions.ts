/** Storybook stand-in for `@/lib/auth/actions`. Those are server actions:
 *  importing the real module drags the database client into the preview
 *  bundle, and stories have no server to call anyway. Every export keeps its
 *  signature and answers as a signed-out browser would. */
import type { User } from "../../src/lib/auth/types";

export interface SessionValidationResult {
  user: User | null;
  isValid: boolean;
  redirectPath?: string;
}

export type ServerLogoutOptions = { globalLogout?: never };

export async function validateSession(
  _currentPath: string,
): Promise<SessionValidationResult> {
  return { user: null, isValid: false };
}

export async function getCurrentUser(): Promise<{
  user: User | null;
  error?: string;
}> {
  return { user: null };
}

export async function getWebSocketToken(): Promise<{
  token: string | null;
  error?: string;
}> {
  return { token: null };
}

export async function serverLogout(_options: ServerLogoutOptions = {}) {
  return { success: true };
}

export async function refreshSession() {
  return { user: null, error: "No active session" };
}
