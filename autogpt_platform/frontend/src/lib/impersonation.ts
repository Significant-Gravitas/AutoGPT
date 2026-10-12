/**
 * Centralized admin impersonation utilities
 * Handles reading, writing, and managing impersonation state across tabs and server/client contexts
 */

import {
  IMPERSONATION_COOKIE_NAME,
  IMPERSONATION_HEADER_NAME,
  IMPERSONATION_OWNER_COOKIE_NAME,
  IMPERSONATION_STORAGE_KEY,
} from "./constants";
import { environment } from "@/services/environment";
import { Key, storage } from "@/services/storage/local-storage";

function writeCookie(name: string, value: string): void {
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; SameSite=Lax; Secure`;
}

function expireCookie(name: string): void {
  document.cookie = `${name}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax; Secure`;
}

function readCookie(name: string): string | null {
  try {
    const cookieValue = document.cookie
      .split("; ")
      .find((row) => row.startsWith(`${name}=`))
      ?.split("=")[1];
    return cookieValue ? decodeURIComponent(cookieValue) : null;
  } catch (error) {
    console.debug("Failed to read impersonation cookie:", error);
    return null;
  }
}

async function readServerCookie(name: string): Promise<string | null> {
  try {
    const { cookies } = await import("next/headers");
    const cookieStore = await cookies();
    return cookieStore.get(name)?.value || null;
  } catch (error) {
    console.debug("Could not access server-side cookies:", error);
    return null;
  }
}

/**
 * Cookie utility functions. The target cookie travels with an owner cookie
 * naming the admin who chose it, so a tab that picks the target up from the
 * cookie also knows whose it is.
 */
export const ImpersonationCookie = {
  set(userId: string, ownerId: string): void {
    if (!environment.isClientSide()) return;
    writeCookie(IMPERSONATION_COOKIE_NAME, userId);
    writeCookie(IMPERSONATION_OWNER_COOKIE_NAME, ownerId);
  },

  clear(): void {
    if (!environment.isClientSide()) return;
    expireCookie(IMPERSONATION_COOKIE_NAME);
    expireCookie(IMPERSONATION_OWNER_COOKIE_NAME);
  },

  get(): string | null {
    if (!environment.isClientSide()) return null;
    return readCookie(IMPERSONATION_COOKIE_NAME);
  },

  getOwner(): string | null {
    if (!environment.isClientSide()) return null;
    return readCookie(IMPERSONATION_OWNER_COOKIE_NAME);
  },

  /**
   * Read the target for server-side requests (Next.js cookies API). A target
   * without an owner cookie is never used.
   */
  async getServerSide(): Promise<string | null> {
    if (environment.isClientSide()) return null;
    const [target, owner] = await Promise.all([
      readServerCookie(IMPERSONATION_COOKIE_NAME),
      readServerCookie(IMPERSONATION_OWNER_COOKIE_NAME),
    ]);
    return target && owner ? target : null;
  },
};

const IMPERSONATION_OWNER_STORAGE_KEY = `${IMPERSONATION_STORAGE_KEY}-owner`;

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

  setOwner(ownerId: string): void {
    if (!environment.isClientSide()) return;

    try {
      sessionStorage.setItem(IMPERSONATION_OWNER_STORAGE_KEY, ownerId);
    } catch (error) {
      console.error("Failed to record the impersonating admin:", error);
    }
  },

  getOwner(): string | null {
    if (!environment.isClientSide()) return null;

    try {
      return sessionStorage.getItem(IMPERSONATION_OWNER_STORAGE_KEY);
    } catch {
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
      sessionStorage.removeItem(IMPERSONATION_OWNER_STORAGE_KEY);
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
/**
 * The target and the admin who chose it, always read from the same place:
 * this tab's sessionStorage if it holds a target, otherwise the cookies. An
 * owner is never borrowed from the other source.
 */
function readImpersonation(): { target: string; owner: string | null } | null {
  const sessionTarget = ImpersonationSession.get();
  if (sessionTarget) {
    return { target: sessionTarget, owner: ImpersonationSession.getOwner() };
  }
  const cookieTarget = ImpersonationCookie.get();
  if (cookieTarget) {
    return { target: cookieTarget, owner: ImpersonationCookie.getOwner() };
  }
  return null;
}

export const ImpersonationState = {
  /**
   * Get the current impersonation target, falling back to the cookie (and
   * adopting its owner) for tabs that didn't start it. A target that can't be
   * tied to the admin who chose it is dropped, never sent.
   */
  get(): string | null {
    const stored = readImpersonation();
    if (!stored) return null;

    if (!stored.owner) {
      ImpersonationState.clear();
      return null;
    }
    if (!ImpersonationSession.get()) {
      ImpersonationSession.set(stored.target);
      ImpersonationSession.setOwner(stored.owner);
    }
    return stored.target;
  },

  /** The admin who chose the current target, from the same source as it. */
  getOwner(): string | null {
    return readImpersonation()?.owner ?? null;
  },

  /**
   * Set the impersonation target, recorded against the admin who chose it,
   * in both sessionStorage and cookies.
   */
  set(userId: string, ownerId: string): void {
    ImpersonationSession.set(userId);
    ImpersonationSession.setOwner(ownerId);
    ImpersonationCookie.set(userId, ownerId);
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

/**
 * Drops impersonation in this tab and tells every other tab to drop theirs.
 * The cookie is shared, but each tab keeps its own sessionStorage copy, which
 * would otherwise keep sending the header for a session that no longer owns
 * it.
 */
export function clearImpersonationEverywhere(): void {
  ImpersonationState.clear();
  storage.set(Key.IMPERSONATION_CLEARED, Date.now().toString());
}

/** Whether any impersonation target is stored, without reading it. */
export function hasImpersonation(): boolean {
  return Boolean(ImpersonationSession.get() ?? ImpersonationCookie.get());
}

export function isImpersonationClearedEvent(event: StorageEvent): boolean {
  return event.key === Key.IMPERSONATION_CLEARED;
}

/**
 * Whether this tab's impersonation target may be used while `user` is signed
 * in: only an admin can impersonate, and only the admin recorded as having
 * chosen the target. No recorded owner means no one may use it.
 */
export function isImpersonationUsableBy(user: {
  id: string;
  role?: string;
}): boolean {
  if (user.role !== "admin") return false;
  return ImpersonationState.getOwner() === user.id;
}

/**
 * Returns system headers to attach to every backend request.
 *
 * Currently adds the impersonation header (X-Act-As-User-Id) when an admin
 * is impersonating a user. Extend this function to add other cross-cutting
 * request headers rather than scattering them across callers.
 *
 * @remarks Client-side only — returns `{}` in server components where
 * `ImpersonationState.get()` has no access to sessionStorage or client cookies.
 * For server-side impersonation, use `ImpersonationState.getServerSide()` instead.
 */
export function getSystemHeaders(): Record<string, string> {
  const headers: Record<string, string> = {};
  const impersonatedUserId = ImpersonationState.get();
  if (impersonatedUserId) {
    headers[IMPERSONATION_HEADER_NAME] = impersonatedUserId;
  }
  return headers;
}
