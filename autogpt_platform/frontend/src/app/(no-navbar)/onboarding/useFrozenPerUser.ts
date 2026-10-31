import { useState } from "react";

/**
 * What `read` returns on the first render where `ready` is true, kept for as
 * long as `userID` stays the same, so a later change to what it reads (a flag
 * answering after its timeout) never reaches the caller. Null until then.
 *
 * Held in state set during render, React's pattern for storing information
 * from previous renders:
 * https://react.dev/reference/react/useState#storing-information-from-previous-renders
 */
export function useFrozenPerUser<T>(
  userID: string | null,
  ready: boolean,
  read: () => T,
): T | null {
  const [frozen, setFrozen] = useState<{
    userID: string | null;
    value: T;
  } | null>(null);
  if (frozen?.userID === userID) return frozen.value;
  if (!ready) return null;
  const value = read();
  setFrozen({ userID, value });
  return value;
}
