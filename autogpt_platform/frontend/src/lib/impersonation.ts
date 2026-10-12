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

/**
 * An impersonation target and the admin who chose it. They are always stored
 * together as one value, so a target can never be read back with an owner
 * that was written for another target.
 */
export interface ImpersonationRecord {
  target: string;
  owner: string;
}

export function encodeImpersonation(record: ImpersonationRecord): string {
  return JSON.stringify({ target: record.target, owner: record.owner });
}

export function decodeImpersonation(
  raw: string | null | undefined,
): ImpersonationRecord | null {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as { target?: unknown; owner?: unknown };
    if (
      typeof value.target === "string" &&
      value.target &&
      typeof value.owner === "string" &&
      value.owner
    ) {
      return { target: value.target, owner: value.owner };
    }
  } catch {
    // A legacy bare id or a corrupted value: no owner, so not usable.
  }
  return null;
}

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

export interface ImpersonationViewer {
  id: string;
  role?: string | null;
}

// This module is in the browser bundle graph, so it can't import the auth
// server instance; it asks Next for the session over HTTP, as
// getServerAuthToken does. Route handlers pass an in-process reader instead.
async function readServerSessionUser(): Promise<ImpersonationViewer | null> {
  try {
    const { cookies } = await import("next/headers");
    const cookieStore = await cookies();
    const cookieHeader = cookieStore
      .getAll()
      .map(({ name, value }) => `${name}=${encodeURIComponent(value)}`)
      .join("; ");
    const baseURL =
      process.env.BETTER_AUTH_INTERNAL_URL ||
      process.env.BETTER_AUTH_URL ||
      process.env.NEXT_PUBLIC_FRONTEND_BASE_URL ||
      "http://localhost:3000";
    const response = await fetch(`${baseURL}/api/auth/get-session`, {
      headers: { cookie: cookieHeader },
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) return null;
    const session = (await response.json()) as {
      user?: { id?: string; role?: string | null };
    } | null;
    return session?.user?.id
      ? { id: session.user.id, role: session.user.role }
      : null;
  } catch (error) {
    console.debug("Could not read the server session:", error);
    return null;
  }
}

function isOwnedBy(
  record: ImpersonationRecord,
  viewer: ImpersonationViewer | null,
): boolean {
  return viewer?.role === "admin" && viewer.id === record.owner;
}

/**
 * Cookie utility functions. The cookie lets other tabs (and server-side
 * requests) pick the target up; it holds the whole record.
 */
export const ImpersonationCookie = {
  set(record: ImpersonationRecord): void {
    if (!environment.isClientSide()) return;
    writeCookie(IMPERSONATION_COOKIE_NAME, encodeImpersonation(record));
  },

  clear(): void {
    if (!environment.isClientSide()) return;
    expireCookie(IMPERSONATION_COOKIE_NAME);
    // Written separately by an earlier version; never read any more.
    expireCookie(IMPERSONATION_OWNER_COOKIE_NAME);
  },

  getRaw(): string | null {
    if (!environment.isClientSide()) return null;
    return readCookie(IMPERSONATION_COOKIE_NAME);
  },

  get(): ImpersonationRecord | null {
    return decodeImpersonation(ImpersonationCookie.getRaw());
  },

  /**
   * The target for a server-side request, only when the session making it is
   * the admin who chose it.
   */
  async getServerSide(
    readCookieValue: (
      name: string,
    ) => Promise<string | null> = readServerCookie,
    readUser: () => Promise<ImpersonationViewer | null> = readServerSessionUser,
  ): Promise<string | null> {
    if (environment.isClientSide()) return null;
    const record = decodeImpersonation(
      await readCookieValue(IMPERSONATION_COOKIE_NAME),
    );
    if (!record) return null;
    return isOwnedBy(record, await readUser()) ? record.target : null;
  },
};

/**
 * SessionStorage utility functions: this tab's own copy of the record.
 */
export const ImpersonationSession = {
  set(record: ImpersonationRecord): void {
    if (!environment.isClientSide()) return;

    try {
      sessionStorage.setItem(
        IMPERSONATION_STORAGE_KEY,
        encodeImpersonation(record),
      );
    } catch (error) {
      console.error("Failed to set impersonation in sessionStorage:", error);
    }
  },

  getRaw(): string | null {
    if (!environment.isClientSide()) return null;

    try {
      return sessionStorage.getItem(IMPERSONATION_STORAGE_KEY);
    } catch (error) {
      console.error("Failed to read impersonation from sessionStorage:", error);
      return null;
    }
  },

  get(): ImpersonationRecord | null {
    return decodeImpersonation(ImpersonationSession.getRaw());
  },

  clear(): void {
    if (!environment.isClientSide()) return;

    try {
      sessionStorage.removeItem(IMPERSONATION_STORAGE_KEY);
      // Written separately by an earlier version; never read any more.
      sessionStorage.removeItem(`${IMPERSONATION_STORAGE_KEY}-owner`);
    } catch (error) {
      console.error(
        "Failed to clear impersonation from sessionStorage:",
        error,
      );
    }
  },
};

// The signed-in user as the auth store last saw them. Headers are only built
// for this viewer, so a target is never sent by anyone but its owner. Browser
// only: on the server this module is shared by every request.
let currentViewer: ImpersonationViewer | null = null;

export function setImpersonationViewer(
  viewer: ImpersonationViewer | null,
): void {
  currentViewer = viewer ? { id: viewer.id, role: viewer.role } : null;
}

/**
 * The record in force: the shared cookie's, mirrored into this tab's copy. A
 * stored value that isn't a whole record is dropped.
 */
function readImpersonation(): ImpersonationRecord | null {
  // The shared cookie is the record of truth: it's what the server checks
  // every request against. This tab's copy only mirrors it, so a copy the
  // cookie doesn't back (cleared, replaced by another tab, or restored from
  // an old session) is dropped rather than sent.
  const cookieRaw = ImpersonationCookie.getRaw();
  const cookieRecord = decodeImpersonation(cookieRaw);
  if (!cookieRecord) {
    if (cookieRaw || ImpersonationSession.getRaw()) ImpersonationState.clear();
    return null;
  }

  const sessionRecord = ImpersonationSession.get();
  if (!sessionRecord || !sameRecord(sessionRecord, cookieRecord)) {
    ImpersonationSession.set(cookieRecord);
  }
  return cookieRecord;
}

function sameRecord(a: ImpersonationRecord, b: ImpersonationRecord): boolean {
  return a.target === b.target && a.owner === b.owner;
}

export const ImpersonationState = {
  /** The stored record, whoever it belongs to. */
  read(): ImpersonationRecord | null {
    return readImpersonation();
  },

  /**
   * The user to act as, only while the signed-in user is the admin who chose
   * them. Anyone else gets null and their requests go out as themselves.
   */
  get(): string | null {
    const record = readImpersonation();
    if (!record) return null;
    return isOwnedBy(record, currentViewer) ? record.target : null;
  },

  /** The admin who chose the stored target, if there is one. */
  getOwner(): string | null {
    return readImpersonation()?.owner ?? null;
  },

  set(userId: string, ownerId: string): void {
    const record = { target: userId, owner: ownerId };
    ImpersonationSession.set(record);
    ImpersonationCookie.set(record);
    // Other tabs are showing (and acting as) the previous choice.
    storage.set(
      Key.IMPERSONATION_CHANGED,
      `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    );
  },

  clear(): void {
    ImpersonationSession.clear();
    ImpersonationCookie.clear();
  },

  async getServerSide(
    readCookieValue?: (name: string) => Promise<string | null>,
    readUser?: () => Promise<ImpersonationViewer | null>,
  ): Promise<string | null> {
    return await ImpersonationCookie.getServerSide(readCookieValue, readUser);
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
  // Unique, so a second clear in the same millisecond still reaches tabs.
  storage.set(
    Key.IMPERSONATION_CLEARED,
    `${Date.now()}-${Math.random().toString(36).slice(2)}`,
  );
}

/** Whether anything is stored, without interpreting it. */
export function hasImpersonation(): boolean {
  return Boolean(ImpersonationSession.getRaw() ?? ImpersonationCookie.getRaw());
}

export function isImpersonationClearedEvent(event: StorageEvent): boolean {
  return event.key === Key.IMPERSONATION_CLEARED;
}

export function isImpersonationChangedEvent(event: StorageEvent): boolean {
  return event.key === Key.IMPERSONATION_CHANGED;
}

/**
 * Whether this tab's impersonation target may be used while `user` is signed
 * in: only an admin can impersonate, and only the admin recorded as having
 * chosen the target. No recorded owner means no one may use it.
 */
export function isImpersonationUsableBy(user: ImpersonationViewer): boolean {
  const record = readImpersonation();
  return record !== null && isOwnedBy(record, user);
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
