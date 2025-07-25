import * as Sentry from "@sentry/nextjs";

export enum Key {
  LOGOUT = "supabase-logout",
  WEBSOCKET_DISCONNECT_INTENT = "websocket-disconnect-intent",
  COPIED_FLOW_DATA = "copied-flow-data",
  SHEPHERD_TOUR = "shepherd-tour",
}

function get(key: Key) {
  if (typeof window === "undefined") {
    Sentry.captureException(new Error("Local storage is not available"));
    return;
  }
  try {
    return window.localStorage.getItem(key);
  } catch {
    // Fine, just return undefined not always items will be set on local storage
    return;
  }
}

function set(key: Key, value: string) {
  if (typeof window === "undefined") {
    Sentry.captureException(new Error("Local storage is not available"));
    return;
  }
  return window.localStorage.setItem(key, value);
}

function clean(key: Key) {
  if (typeof window === "undefined") {
    Sentry.captureException(new Error("Local storage is not available"));
    return;
  }
  return window.localStorage.removeItem(key);
}

export const storage = {
  clean,
  get,
  set,
};
