"use client";

import { useBuilderSidebarAutoClose } from "./useBuilderSidebarAutoClose";

// The builder needs the full canvas, so entering /build always starts with the
// sidebar collapsed — the user can still re-open it manually. Leaving /build
// restores whatever state the sidebar was in before.
export function BuilderSidebarAutoClose() {
  useBuilderSidebarAutoClose();

  return null;
}
