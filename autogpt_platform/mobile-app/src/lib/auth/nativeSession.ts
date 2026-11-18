import * as SecureStore from "expo-secure-store";
import { useSyncExternalStore } from "react";

import { FRONTEND_URL } from "@/lib/env";

import {
  NO_APP_SESSION_CODE,
  parseStoredSession,
  readSignInFailure,
  readUser,
  sessionCookieHeader,
  storageKey,
  type SignInFailure,
  type StoredSession,
} from "./sessionHelpers";

// The auth service's contract for installed apps
// (docs/engineering/native-mobile/backend-handoff.md): sign in with this
// marker and no cookie or Origin, keep the signed credential it answers with,
// and send that as a bearer to the auth service only.
const NATIVE_MARKER = { "X-AutoGPT-Native": "1" };
const AUTH_URL = `${FRONTEND_URL}/api/auth`;
const STORAGE_KEY = storageKey(FRONTEND_URL);

let current = readSaved();
const listeners = new Set<() => void>();

function readSaved() {
  try {
    return parseStoredSession(SecureStore.getItem(STORAGE_KEY));
  } catch {
    return null;
  }
}

function setCurrent(next: StoredSession | null) {
  current = next;
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** The signed-in session, read from the phone at launch, or null. */
export function useSession() {
  return useSyncExternalStore(subscribe, () => current);
}

export function hasSession() {
  return current !== null;
}

/** The signed credential, which names the session: a new sign-in gets a new one. */
export function sessionToken() {
  return current?.token ?? null;
}

/** The session as a Cookie header for the web app's own pages and routes. */
export function getSessionCookie() {
  return current ? sessionCookieHeader(current.token, FRONTEND_URL) : null;
}

/** Resolves to null once signed in, or to why the server refused. Throws when it can't be reached. */
export async function signInWithEmail(
  email: string,
  password: string,
): Promise<SignInFailure | null> {
  const response = await send("/sign-in/email", null, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!response.ok) return readSignInFailure(response);
  const token = response.headers.get("set-auth-token");
  const user = readUser(await response.json().catch(() => null));
  // A server without the installed-app contract answers with a cookie only.
  if (!token || !user) {
    return { status: response.status, code: NO_APP_SESSION_CODE };
  }
  const session = { token, user };
  SecureStore.setItem(STORAGE_KEY, JSON.stringify(session));
  setCurrent(session);
  return null;
}

/** A request to the auth service as the signed-in user. */
export function authRequest(path: string, init: RequestInit = {}) {
  return send(path, current, init);
}

/**
 * Asks the auth service whether the saved session still stands. False only
 * on a clear no: a server that can't be reached leaves the person signed in.
 */
export async function confirmSession() {
  const session = current;
  if (!session) return true;
  let body: unknown;
  try {
    const response = await send("/get-session", session);
    if (!response.ok) return true;
    body = await response.json();
  } catch {
    return true;
  }
  // Someone signed out, or in again, while the answer was on its way.
  if (current !== session) return true;
  if (body === null) return false;
  const user = readUser(body);
  if (user) {
    const next = { token: session.token, user };
    SecureStore.setItem(STORAGE_KEY, JSON.stringify(next));
    setCurrent(next);
  }
  return true;
}

/**
 * Ends the session on the phone first, so the app is signed out even when
 * the server can't be told.
 */
export async function endSession(signal?: AbortSignal) {
  const session = current;
  setCurrent(null);
  await SecureStore.deleteItemAsync(STORAGE_KEY).catch(() => undefined);
  if (!session) return;
  await send("/sign-out", session, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    signal,
  });
}

function send(
  path: string,
  session: StoredSession | null,
  init: RequestInit = {},
) {
  return fetch(`${AUTH_URL}${path}`, {
    ...init,
    headers: {
      ...NATIVE_MARKER,
      ...(session ? { Authorization: `Bearer ${session.token}` } : {}),
      ...(init.headers as Record<string, string> | undefined),
    },
    // The phone's cookie store would add a Cookie header, and the auth
    // service then treats the request as a browser's.
    credentials: "omit",
  });
}
