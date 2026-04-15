import * as Sentry from "@sentry/nextjs";
import { environment } from "../environment";

export enum SessionKey {
  CHAT_SENT_INITIAL_PROMPTS = "chat_sent_initial_prompts",
  CHAT_INITIAL_PROMPTS = "chat_initial_prompts",
}

function get(key: SessionKey) {
  if (environment.isServerSide()) {
    Sentry.captureException(new Error("Session storage is not available"));
    return;
  }
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return;
  }
}

function set(key: SessionKey, value: string) {
  if (environment.isServerSide()) {
    Sentry.captureException(new Error("Session storage is not available"));
    return;
  }
  return window.sessionStorage.setItem(key, value);
}

function clean(key: SessionKey) {
  if (environment.isServerSide()) {
    Sentry.captureException(new Error("Session storage is not available"));
    return;
  }
  return window.sessionStorage.removeItem(key);
}

export const sessionStorage = {
  clean,
  get,
  set,
};
