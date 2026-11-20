"use client";

import { environment } from "@/services/environment";
import { useSyncExternalStore } from "react";
import { localExecutorDeployment } from "./helpers";

function subscribeToNothing() {
  return () => {};
}

function readOrigin() {
  return window.location.origin;
}

function readNoOrigin() {
  return null;
}

export function useLocalExecutorSetup() {
  // The pairing commands name this page's origin, which the server render
  // cannot know; it renders without them and the browser fills them in.
  const origin = useSyncExternalStore(
    subscribeToNothing,
    readOrigin,
    readNoOrigin,
  );
  const deployment = origin
    ? localExecutorDeployment(environment.getAGPTServerApiUrl(), origin)
    : null;

  return { deployment };
}
