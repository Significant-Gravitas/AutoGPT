import * as SecureStore from "expo-secure-store";

import { parsePendingFlow, type PendingFlow } from "./oauthHelpers";

// One slot: only one system sign-in sheet can be open at a time.
const STORAGE_KEY = "connections.pending-oauth";

// Saves and clears run one at a time, so a clear for an older sign-in can't
// read that flow and then delete a newer one saved in between.
let queue: Promise<unknown> = Promise.resolve();

function inTurn<T>(task: () => Promise<T>) {
  const run = queue.then(task, task);
  queue = run.catch(() => undefined);
  return run;
}

export function savePendingFlow(flow: PendingFlow) {
  return inTurn(() => SecureStore.setItemAsync(STORAGE_KEY, JSON.stringify(flow)));
}

/** The saved flow for `state`, if it is still fresh. */
export async function readPendingFlow(state: string) {
  const raw = await SecureStore.getItemAsync(STORAGE_KEY).catch(() => null);
  const flow = parsePendingFlow(raw, Date.now());
  return flow?.state === state ? flow : null;
}

/** Forgets the saved flow, but only if it is still this one. */
export function clearPendingFlow(state: string) {
  return inTurn(async () => {
    const raw = await SecureStore.getItemAsync(STORAGE_KEY).catch(() => null);
    const flow = parsePendingFlow(raw, Date.now());
    if (raw && (!flow || flow.state === state)) {
      await SecureStore.deleteItemAsync(STORAGE_KEY).catch(() => undefined);
    }
  });
}
