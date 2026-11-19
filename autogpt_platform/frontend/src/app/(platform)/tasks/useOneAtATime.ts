import { useRef } from "react";

/** Runs an action only while the same action is not already in flight,
 *  checked at once rather than after a re-render, so a double click or a
 *  second Enter in the same moment sends one request. `start` gets a
 *  `done` to call when the request settles. */
export function useOneAtATime() {
  const busy = useRef(new Set<string>());

  return function once(key: string, start: (done: () => void) => void) {
    if (busy.current.has(key)) return;
    busy.current.add(key);
    start(() => busy.current.delete(key));
  };
}
