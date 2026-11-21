import { useSyncExternalStore } from "react";

import type { Expert } from "../../lib/api/experts";

// Shared by the list and the details page: a hire started on one shows, and
// can't be sent again, on the other. Everything belongs to one sign-in
// (`generation`), so another account starts clean and a late answer for the
// last one can't touch its state.
export interface HireState {
  generation: number;
  inFlight: ReadonlySet<string>;
  /** Hires confirmed in this sign-in, by template id. */
  hired: ReadonlyMap<string, Expert>;
  /** When each of those was confirmed. */
  hiredAt: ReadonlyMap<string, number>;
}

let state: HireState = { generation: 0, inFlight: new Set(), hired: new Map(), hiredAt: new Map() };
const listeners = new Set<() => void>();

function publish(next: HireState) {
  state = next;
  listeners.forEach((listener) => listener());
}

function stateFor(generation: number): HireState {
  if (state.generation === generation) return state;
  return { generation, inFlight: new Set(), hired: new Map(), hiredAt: new Map() };
}

/** Marks a hire as started; false when that template's hire is already running. */
export function claimHire(templateID: string, generation: number) {
  const current = stateFor(generation);
  if (current.inFlight.has(templateID)) return false;
  publish({ ...current, inFlight: new Set([...current.inFlight, templateID]) });
  return true;
}

export function releaseHire(templateID: string, generation: number) {
  if (state.generation !== generation || !state.inFlight.has(templateID)) return;
  const inFlight = new Set(state.inFlight);
  inFlight.delete(templateID);
  publish({ ...state, inFlight });
}

export function recordHire(templateID: string, expert: Expert, generation: number) {
  if (state.generation !== generation) return;
  publish({
    ...state,
    hired: new Map([...state.hired, [templateID, expert]]),
    hiredAt: new Map([...state.hiredAt, [templateID, Date.now()]]),
  });
}

/**
 * Hires newer than the team list's last load (`dataUpdatedAt`), which the
 * Marketplace adds to it. Once the list has loaded after a hire it decides,
 * so an expert removed on the web stays gone; hiring again starts afresh.
 */
export function hiresSince(current: HireState, listUpdatedAt: number) {
  return new Map(
    [...current.hired].filter(
      ([templateID]) => (current.hiredAt.get(templateID) ?? 0) >= listUpdatedAt,
    ),
  );
}

/** The hire state of the given sign-in; empty for any other. */
export function hireStateFor(current: HireState, generation: number) {
  return current.generation === generation ? current : stateFor(generation);
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function readHireState() {
  return state;
}

export function useHireStore() {
  return useSyncExternalStore(subscribe, readHireState);
}
