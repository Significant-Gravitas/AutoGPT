"use client";

import { useArtifactsPanelNavCollapse } from "./useArtifactsPanelNavCollapse";

// The artifacts panel needs the horizontal room, so opening it collapses the
// left navigation; closing it restores whatever state the nav had before the
// panel opened — expanded stays expanded, collapsed stays collapsed.
export function ArtifactsPanelNavCollapse() {
  useArtifactsPanelNavCollapse();

  return null;
}
