/**
 * API client for backend authentication endpoints.
 */

import type {
  AuthResult,
  AuthTokens,
  AuthUser,
  LoginCredentials,
  RegisterCredentials,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_AGPT_SERVER_URL || "http://localhost:8006/api";

/**
 * Make an authenticated API request.
 */
async function authFetch<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<AuthResult<T>> {
  try {
    const response = await fetch(`${API_BASE_URL}/auth${endpoint}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return {
        data: null,
        error: {
          message: data.detail || data.message || "An error occurred",
          code: response.status.toString(),
        },
      };
    }

    return { data: data as T, error: null };
  } catch (error) {
    return {
      data: null,
      error: {
        message:
          error instanceof Error ? error.message : "Network error occurred",
        code: "NETWORK_ERROR",
      },
    };
  }
}

/**
 * Register a new user with email and password.
 */
export async function register(
  credentials: RegisterCredentials,
): Promise<AuthResult<AuthTokens>> {
  return authFetch<AuthTokens>("/register", {
    method: "POST",
    body: JSON.stringify(credentials),
  });
}

/**
 * Login with email and password.
 */
export async function login(
  credentials: LoginCredentials,
): Promise<AuthResult<AuthTokens>> {
  return authFetch<AuthTokens>("/login", {
    method: "POST",
    body: JSON.stringify(credentials),
  });
}

/**
 * Logout by revoking the refresh token.
 */
export async function logout(refreshToken: string): Promise<AuthResult<void>> {
  return authFetch<void>("/logout", {
    method: "POST",
    body: JSON.stringify({ refresh_token: refreshToken }),
  });
}

/**
 * Refresh access token using refresh token.
 */
export async function refreshTokens(
  refreshToken: string,
): Promise<AuthResult<AuthTokens>> {
  return authFetch<AuthTokens>("/refresh", {
    method: "POST",
    body: JSON.stringify({ refresh_token: refreshToken }),
  });
}

/**
 * Request password reset email.
 */
export async function requestPasswordReset(
  email: string,
): Promise<AuthResult<{ message: string }>> {
  return authFetch<{ message: string }>("/password-reset/request", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

/**
 * Confirm password reset with token.
 */
export async function confirmPasswordReset(
  token: string,
  newPassword: string,
): Promise<AuthResult<{ message: string }>> {
  return authFetch<{ message: string }>("/password-reset/confirm", {
    method: "POST",
    body: JSON.stringify({ token, new_password: newPassword }),
  });
}

/**
 * Get Google OAuth login URL.
 */
export async function getGoogleLoginUrl(): Promise<
  AuthResult<{ url: string }>
> {
  return authFetch<{ url: string }>("/google/login", {
    method: "GET",
  });
}

/**
 * Decode JWT payload to extract user info.
 * Note: This does NOT verify the token - verification is done server-side.
 */
export function decodeJwtPayload(token: string): AuthUser | null {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;

    const payload = JSON.parse(atob(parts[1]));

    return {
      id: payload.sub,
      email: payload.email || "",
      name: payload.name || null,
      role: payload.role || "authenticated",
      emailVerified: payload.email_verified || false,
    };
  } catch {
    return null;
  }
}
