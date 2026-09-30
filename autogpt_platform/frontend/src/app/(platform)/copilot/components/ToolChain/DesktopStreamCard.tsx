"use client";

import {
  DesktopStreamPreview,
  isDesktopStream,
} from "@/components/contextual/OutputRenderers/renderers/DesktopStreamRenderer";
import { useEffect } from "react";
import { useCopilotUIStore, type DesktopStreamRef } from "../../store";
import { useIsMobile } from "../../useIsMobile";

interface Props {
  stream: unknown;
  /** A shared transcript: the viewer is not the owner and has no panel. */
  readOnly?: boolean;
}

function asStreamRef(value: unknown): DesktopStreamRef | null {
  if (typeof value !== "object" || value === null) return null;
  const v = value as Record<string, unknown>;
  if (typeof v.url !== "string" || typeof v.sandbox_id !== "string")
    return null;
  return {
    url: v.url,
    sandbox_id: v.sandbox_id,
    provider: typeof v.provider === "string" ? v.provider : "e2b",
  };
}

/** The inline start_desktop card. Besides embedding the stream it tells the
 *  side panel a desktop exists, so the Computer face can show the same
 *  screen without the model being asked again. */
export function DesktopStreamCard({ stream, readOnly = false }: Props) {
  const registerComputerStream = useCopilotUIStore(
    (s) => s.registerComputerStream,
  );
  const isMobile = useIsMobile();
  const ref = asStreamRef(stream);
  const sandboxId = ref?.sandbox_id;
  const url = ref?.url;
  useEffect(() => {
    // The mobile sheet has no computer face: remember the desktop, open nothing.
    if (ref && !readOnly) registerComputerStream(ref, { show: !isMobile });
    // Re-register only when the stream itself changes, not on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sandboxId, url, readOnly, isMobile]);
  if (!isDesktopStream(stream)) return null;
  return <DesktopStreamPreview value={stream} ownerView={!readOnly} />;
}
