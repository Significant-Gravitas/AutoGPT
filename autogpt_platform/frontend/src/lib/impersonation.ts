/**
 * Centralized admin impersonation utilities
 * Handles reading, writing, and managing impersonation state across tabs and server/client contexts
 */

import { IMPERSONATION_STORAGE_KEY } from "./constants";
import { environment } from "@/services/environment";

const COOKIE_NAME = "admin-impersonate-user-id";

/**
 * Cookie utility functions
 */
export const ImpersonationCookie = {
  /**
   * Set impersonation cookie with proper security attributes
   */
  set(userId: string): void {
    if (!environment.isClientSide()) return;

    const encodedUserId = encodeURIComponent(userId);
    document.cookie = `${COOKIE_NAME}=${encodedUserId}; path=/; SameSite=Lax; Secure`;
  },

  /**
   * Clear impersonation cookie
   */
  clear(): void {
    if (!environment.isClientSide()) return;

    document.cookie = `${COOKIE_NAME}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax; Secure`;
  },

  /**
   * Read impersonation cookie (client-side)
   */
  get(): string | null {
    if (!environment.isClientSide()) return null;

    try {
      const cookieValue = document.cookie
        .split("; ")
        .find((row) => row.startsWith(`${COOKIE_NAME}=`))
        ?.split("=")[1];

      return cookieValue ? decodeURIComponent(cookieValue) : null;
    } catch (error) {
      console.debug("Failed to read impersonation cookie:", error);
      return null;
    }
  },

  /**
   * Read impersonation cookie (server-side using Next.js cookies API)
   */
  async getServerSide(): Promise<string | null> {
    if (environment.isClientSide()) return null;

    try {
      const { cookies } = await import("next/headers");
      const cookieStore = await cookies();
      const impersonationCookie = cookieStore.get(COOKIE_NAME);
      return impersonationCookie?.value || null;
    } catch (error) {
      console.debug("Could not access server-side cookies:", error);
      return null;
    }
  },
};

/**
 * SessionStorage utility functions
 */
export const ImpersonationSession = {
  /**
   * Set impersonation in sessionStorage
   */
  set(userId: string): void {
    if (!environment.isClientSide()) return;

    try {
      sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, userId);
    } catch (error) {
      console.error("Failed to set impersonation in sessionStorage:", error);
    }
  },

  /**
   * Get impersonation from sessionStorage
   */
  get(): string | null {
    if (!environment.isClientSide()) return null;

    try {
      return sessionStorage.getItem(IMPERSONATION_STORAGE_KEY);
    } catch (error) {
      console.error("Failed to read impersonation from sessionStorage:", error);
      return null;
    }
  },

  /**
   * Clear impersonation from sessionStorage
   */
  clear(): void {
    if (!environment.isClientSide()) return;

    try {
      sessionStorage.removeItem(IMPERSONATION_STORAGE_KEY);
    } catch (error) {
      console.error(
        "Failed to clear impersonation from sessionStorage:",
        error,
      );
    }
  },
};

/**
 * Main impersonation state management
 */
export const ImpersonationState = {
  /**
   * Get current impersonation user ID with cross-tab fallback
   * Checks sessionStorage first, then falls back to cookie for cross-tab support
   */
  get(): string | null {
    // First check sessionStorage (same tab)
    const sessionValue = ImpersonationSession.get();
    if (sessionValue) {
      return sessionValue;
    }

    // Fallback to cookie (cross-tab support)
    const cookieValue = ImpersonationCookie.get();
    if (cookieValue) {
      // Sync back to sessionStorage for consistency
      ImpersonationSession.set(cookieValue);
      return cookieValue;
    }

    return null;
  },

  /**
   * Set impersonation user ID in both sessionStorage and cookie
   */
  set(userId: string): void {
    ImpersonationSession.set(userId);
    ImpersonationCookie.set(userId);
  },

  /**
   * Clear impersonation from both sessionStorage and cookie
   */
  clear(): void {
    ImpersonationSession.clear();
    ImpersonationCookie.clear();
  },

  /**
   * Get impersonation user ID for server-side requests
   */
  async getServerSide(): Promise<string | null> {
    return await ImpersonationCookie.getServerSide();
  },
};
