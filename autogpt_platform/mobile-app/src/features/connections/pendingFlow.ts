import * as SecureStore from "expo-secure-store";

import { parsePendingFlow, type PendingFlow } from "./oauthHelpers";

// One slot: only one system sign-in sheet can be open at a time.
const STORAGE_KEY = "connections.pending-oauth";

export async function savePendingFlow(flow: PendingFlow) {
  await SecureStore.setItemAsync(STORAGE_KEY, JSON.stringify(flow));
}

/** The saved flow for `state`, if it is still fresh. */
export async function readPendingFlow(state: string) {
  const raw = await SecureStore.getItemAsync(STORAGE_KEY).catch(() => null);
  const flow = parsePendingFlow(raw, Date.now());
  return flow?.state === state ? flow : null;
}

/** Forgets the saved flow, but only if it is still this one. */
export async function clearPendingFlow(state: string) {
  const raw = await SecureStore.getItemAsync(STORAGE_KEY).catch(() => null);
  const flow = parsePendingFlow(raw, Date.now());
  if (raw && (!flow || flow.state === state)) {
    await SecureStore.deleteItemAsync(STORAGE_KEY).catch(() => undefined);
  }
}
