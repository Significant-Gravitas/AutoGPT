import { useSyncExternalStore } from "react";
import { chatLinkTarget } from "./board-helpers";

function subscribe() {
  return () => {};
}

function readTarget() {
  return chatLinkTarget(window.location.search, window.location.hash);
}

// The hash never reaches the server, so the answer is unknown (undefined)
// until the page runs in the browser.
function readServerTarget() {
  return undefined;
}

export function useChatLinkTarget() {
  return useSyncExternalStore(subscribe, readTarget, readServerTarget);
}
