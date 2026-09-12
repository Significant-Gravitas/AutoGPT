"use client";

import { desktopStreamRenderer } from "@/components/contextual/OutputRenderers/renderers/DesktopStreamRenderer";
import { useEffect } from "react";
import { useCopilotUIStore, type DesktopStreamRef } from "../../store";

interface Props {
  stream: unknown;
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
export function DesktopStreamCard({ stream }: Props) {
  const registerComputerStream = useCopilotUIStore(
    (s) => s.registerComputerStream,
  );
  const ref = asStreamRef(stream);
  const sandboxId = ref?.sandbox_id;
  const url = ref?.url;
  useEffect(() => {
    if (ref) registerComputerStream(ref);
    // Re-register only when the stream itself changes, not on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sandboxId, url]);
  return <>{desktopStreamRenderer.render(stream)}</>;
}
