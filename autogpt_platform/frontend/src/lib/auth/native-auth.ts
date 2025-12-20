/**
 * Native authentication client for FastAPI backend.
 *
 * This module provides authentication functions that communicate with the
 * FastAPI backend instead of Supabase, while maintaining interface compatibility.
 */

import { environment } from "@/services/environment";

// User type compatible with Supabase User for minimal frontend changes
export interface User {
  id: string;
  email: string;
  email_verified: boolean;
  name?: string;
  created_at: string;
  role?: string;
}

export interface AuthResponse {
  user: User;
  access_token: string;
  refresh_token: string;
  expires_in: number;
  token_type: string;
}

export interface MessageResponse {
  message: string;
}

// Cookie names (must match backend)
const ACCESS_TOKEN_COOKIE = "access_token";

function getApiUrl(): string {
  return environment.getAGPTServerBaseUrl();
}

/**
 * Parse JWT token without verification (for client-side use only).
 * The backend always verifies tokens - this is just for UI purposes.
 */
export function parseJwt(token: string): Record<string, unknown> | null {
  try {
    const base64Url = token.split(".")[1];
    const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split("")
        .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
        .join(""),
    );
    return JSON.parse(jsonPayload);
  } catch {
    return null;
  }
}

/**
 * Get access token from cookie.
 */
export function getAccessToken(): string | null {
  if (typeof document === "undefined") return null;

  const cookies = document.cookie.split(";");
  for (const cookie of cookies) {
    const [name, value] = cookie.trim().split("=");
    if (name === ACCESS_TOKEN_COOKIE) {
      return value;
    }
  }
  return null;
}

/**
 * Check if user is authenticated based on cookie presence and token validity.
 */
export function isAuthenticated(): boolean {
  const token = getAccessToken();
  if (!token) return false;

  const payload = parseJwt(token);
  if (!payload) return false;

  // Check expiration
  const exp = payload.exp as number;
  if (exp && Date.now() / 1000 > exp) return false;

  return true;
}

/**
 * Get current user from access token (client-side only).
 * For server-side, use the server action.
 */
export function getCurrentUserFromToken(): User | null {
  const token = getAccessToken();
  if (!token) return null;

  const payload = parseJwt(token);
  if (!payload) return null;

  // Check expiration
  const exp = payload.exp as number;
  if (exp && Date.now() / 1000 > exp) return null;

  return {
    id: payload.sub as string,
    email: payload.email as string,
    email_verified: true, // If token is valid, email is verified
    role: payload.role as string,
    created_at: new Date().toISOString(), // Not available in token
  };
}

// ============================================================================
// Server-side API calls (for use in server actions)
// ============================================================================

export async function serverLogin(
  email: string,
  password: string,
): Promise<{ success: boolean; error?: string; user?: User }> {
  try {
    const response = await fetch(`${getApiUrl()}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
      credentials: "include",
    });

    if (!response.ok) {
      const error = await response.json();
      return { success: false, error: error.detail || "Login failed" };
    }

    const data: AuthResponse = await response.json();
    return { success: true, user: data.user };
  } catch (error) {
    console.error("Login error:", error);
    return { success: false, error: "Network error" };
  }
}

export async function serverSignup(
  email: string,
  password: string,
): Promise<{ success: boolean; error?: string; message?: string }> {
  try {
    const response = await fetch(`${getApiUrl()}/api/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
      credentials: "include",
    });

    if (!response.ok) {
      const error = await response.json();
      return { success: false, error: error.detail || "Signup failed" };
    }

    const data: MessageResponse = await response.json();
    return { success: true, message: data.message };
  } catch (error) {
    console.error("Signup error:", error);
    return { success: false, error: "Network error" };
  }
}

export async function serverLogout(
  scope: "local" | "global" = "local",
): Promise<{ success: boolean; error?: string }> {
  try {
    const response = await fetch(`${getApiUrl()}/api/auth/logout?scope=${scope}`, {
      method: "POST",
      credentials: "include",
    });

    if (!response.ok) {
      const error = await response.json();
      return { success: false, error: error.detail || "Logout failed" };
    }

    return { success: true };
  } catch (error) {
    console.error("Logout error:", error);
    return { success: false, error: "Network error" };
  }
}

export async function serverRefreshToken(): Promise<{
  success: boolean;
  error?: string;
  user?: User;
}> {
  try {
    const response = await fetch(`${getApiUrl()}/api/auth/refresh`, {
      method: "POST",
      credentials: "include",
    });

    if (!response.ok) {
      const error = await response.json();
      return { success: false, error: error.detail || "Refresh failed" };
    }

    const data: AuthResponse = await response.json();
    return { success: true, user: data.user };
  } catch (error) {
    console.error("Refresh error:", error);
    return { success: false, error: "Network error" };
  }
}

export async function serverGetCurrentUser(): Promise<{
  user: User | null;
  error?: string;
}> {
  try {
    const response = await fetch(`${getApiUrl()}/api/auth/me`, {
      method: "GET",
      credentials: "include",
    });

    if (!response.ok) {
      if (response.status === 401) {
        return { user: null };
      }
      const error = await response.json();
      return { user: null, error: error.detail || "Failed to get user" };
    }

    const user: User = await response.json();
    return { user };
  } catch (error) {
    console.error("Get current user error:", error);
    return { user: null, error: "Network error" };
  }
}

export async function serverRequestPasswordReset(
  email: string,
): Promise<{ success: boolean; error?: string; message?: string }> {
  try {
    const response = await fetch(`${getApiUrl()}/api/auth/password/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });

    if (!response.ok) {
      const error = await response.json();
      return { success: false, error: error.detail || "Reset request failed" };
    }

    const data: MessageResponse = await response.json();
    return { success: true, message: data.message };
  } catch (error) {
    console.error("Password reset request error:", error);
    return { success: false, error: "Network error" };
  }
}

export async function serverSetPassword(
  token: string,
  password: string,
): Promise<{ success: boolean; error?: string; message?: string }> {
  try {
    const response = await fetch(`${getApiUrl()}/api/auth/password/set`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      return { success: false, error: error.detail || "Password set failed" };
    }

    const data: MessageResponse = await response.json();
    return { success: true, message: data.message };
  } catch (error) {
    console.error("Set password error:", error);
    return { success: false, error: "Network error" };
  }
}

export async function serverGetGoogleAuthUrl(
  redirectTo: string = "/marketplace",
): Promise<{ url?: string; error?: string }> {
  try {
    const response = await fetch(
      `${getApiUrl()}/api/auth/oauth/google/authorize?redirect_to=${encodeURIComponent(redirectTo)}`,
      {
        method: "GET",
        credentials: "include",
      },
    );

    if (!response.ok) {
      const error = await response.json();
      return { error: error.detail || "Failed to get OAuth URL" };
    }

    const data = await response.json();
    return { url: data.url };
  } catch (error) {
    console.error("Get Google auth URL error:", error);
    return { error: "Network error" };
  }
}
