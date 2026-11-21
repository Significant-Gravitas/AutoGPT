"use client";

import { usePostV2StartSessionDesktop } from "@/app/api/__generated__/endpoints/chat/chat";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useStableCallback } from "@/hooks/useStableCallback";
import {
  DesktopStreamPreview,
  isDesktopStream,
} from "@/components/contextual/OutputRenderers/renderers/DesktopStreamRenderer";
import { useContext, useEffect, useState } from "react";
import { desktopStartFailure } from "../ArtifactPanel/components/desktopStartFailure";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
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
    requires_auth: v.requires_auth === true,
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
  const { toast } = useToast();
  const sessionId = useContext(CopilotChatActionsContext)?.sessionId;
  // The tool's stream stops when its turn ends; Reconnect opens a fresh one.
  const [reopened, setReopened] = useState<unknown>(null);
  const current = reopened ?? stream;
  const ref = asStreamRef(current);
  const sandboxId = ref?.sandbox_id;
  const url = ref?.url;
  const registerStream = useStableCallback(() => {
    // The mobile sheet has no computer face: remember the desktop, open nothing.
    if (ref && !readOnly) registerComputerStream(ref, { show: !isMobile });
  });
  // Re-register only when the stream itself changes, not on every render.
  useEffect(() => {
    registerStream();
  }, [sandboxId, url, readOnly, isMobile]);

  const { mutate: startDesktop, isPending } = usePostV2StartSessionDesktop({
    mutation: {
      onSuccess: (res) => {
        const next = okData(res);
        if (next) setReopened({ ...next, kind: "desktop_stream" });
      },
      onError: (error) => {
        toast({
          title: "Could not open the desktop",
          description: desktopStartFailure(error),
          variant: "destructive",
        });
      },
    },
  });

  if (!isDesktopStream(current)) return null;
  return (
    <DesktopStreamPreview
      value={current}
      ownerView={!readOnly}
      onReconnect={
        sessionId && !readOnly ? () => startDesktop({ sessionId }) : undefined
      }
      reconnecting={isPending}
    />
  );
}
