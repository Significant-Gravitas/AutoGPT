import { useSyncExternalStore } from "react";

import type { Expert } from "../../lib/api/experts";

// Shared by the list and the details page: a hire started on one shows, and
// can't be sent again, on the other. Everything belongs to one sign-in
// (`generation`), so another account starts clean and a late answer for the
// last one can't touch its state.
export interface HireState {
  generation: number;
  inFlight: ReadonlySet<string>;
  /** Hires confirmed in this sign-in, by template id, until the team list has them. */
  hired: ReadonlyMap<string, Expert>;
}

let state: HireState = { generation: 0, inFlight: new Set(), hired: new Map() };
const listeners = new Set<() => void>();

function publish(next: HireState) {
  state = next;
  listeners.forEach((listener) => listener());
}

function stateFor(generation: number): HireState {
  if (state.generation === generation) return state;
  return { generation, inFlight: new Set(), hired: new Map() };
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
  publish({ ...state, hired: new Map([...state.hired, [templateID, expert]]) });
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
