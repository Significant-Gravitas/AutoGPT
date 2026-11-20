import { useSyncExternalStore } from "react";

/** An account a row was answered with: signed in, added, picked or allowed. */
export interface PickedCredential {
  id: string;
  title: string | null;
}

export type FollowUpState = "idle" | "sending" | "sent";

/**
 * What happened on one setup card in this app run. Kept outside React so a
 * card scrolled away and back (or rebuilt by a history reload) remembers it,
 * and a chat opened again never sends its follow-up twice.
 */
export interface CardRun {
  picks: Record<string, PickedCredential>;
  followUp: FollowUpState;
  error: string | null;
}

const EMPTY: CardRun = { picks: {}, followUp: "idle", error: null };
const runs = new Map<string, CardRun>();
const listeners = new Set<() => void>();

function update(cardKey: string, change: (run: CardRun) => CardRun) {
  runs.set(cardKey, change(readCardRun(cardKey)));
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function readCardRun(cardKey: string) {
  return runs.get(cardKey) ?? EMPTY;
}

export function useCardRun(cardKey: string) {
  return useSyncExternalStore(subscribe, () => readCardRun(cardKey));
}

export function markConnected(
  cardKey: string,
  rowKey: string,
  credential: PickedCredential,
) {
  update(cardKey, (run) => ({
    ...run,
    picks: { ...run.picks, [rowKey]: credential },
    error: null,
  }));
}

/** True for the one caller that may send this card's follow-up now. */
export function claimFollowUp(cardKey: string) {
  if (readCardRun(cardKey).followUp !== "idle") return false;
  update(cardKey, (run) => ({ ...run, followUp: "sending", error: null }));
  return true;
}

export function settleFollowUp(cardKey: string, isSent: boolean, error: string | null = null) {
  update(cardKey, (run) => ({ ...run, followUp: isSent ? "sent" : "idle", error }));
}
