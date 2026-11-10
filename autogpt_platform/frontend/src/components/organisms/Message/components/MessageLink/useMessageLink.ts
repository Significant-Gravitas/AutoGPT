import { useSyncExternalStore } from "react";
import { kindOfLink } from "./helpers";

function subscribeToNothing() {
  return () => {};
}

function getOrigin() {
  return window.location.origin;
}

function getServerOrigin() {
  return null;
}

export function useMessageLink(href: string | undefined) {
  const origin = useSyncExternalStore(
    subscribeToNothing,
    getOrigin,
    getServerOrigin,
  );
  return { kind: kindOfLink(href, origin) };
}
