"use client";

import {
  getGetV2GetSessionComputerQueryKey,
  useGetV2GetSessionComputer,
  usePostV2StartSessionDesktop,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { okData } from "@/app/api/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { desktopStreamRenderer } from "@/components/contextual/OutputRenderers/renderers/DesktopStreamRenderer";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { cn } from "@/lib/utils";
import { useQueryClient } from "@tanstack/react-query";
import { useCopilotUIStore } from "../../../store";

interface Props {
  sessionId: string;
}

function StatePill({ state }: { state: "running" | "paused" | null }) {
  return (
    <span
      className={cn(
        "rounded-full px-2 py-0.5 text-xs font-medium",
        state === "running"
          ? "bg-emerald-50 text-emerald-700"
          : state === "paused"
            ? "bg-zinc-100 text-zinc-600"
            : "bg-zinc-50 text-zinc-500",
      )}
    >
      {state === "running"
        ? "Running"
        : state === "paused"
          ? "Suspended"
          : "None"}
    </span>
  );
}

export function ComputerPanelContent({ sessionId }: Props) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const stream = useCopilotUIStore((s) => s.artifactPanel.computer);
  const registerComputerStream = useCopilotUIStore(
    (s) => s.registerComputerStream,
  );

  const computerQuery = useGetV2GetSessionComputer(sessionId, {
    query: {
      select: (res) => okData(res) ?? null,
      refetchInterval: 15_000,
    },
  });
  const computer = computerQuery.data ?? null;

  const { mutate: startDesktop, isPending } = usePostV2StartSessionDesktop({
    mutation: {
      onSuccess: (res) => {
        const next = okData(res);
        if (next) {
          registerComputerStream({
            url: next.url,
            sandbox_id: next.sandbox_id,
            provider: next.provider ?? "e2b",
          });
        }
        queryClient.invalidateQueries({
          queryKey: getGetV2GetSessionComputerQueryKey(sessionId),
        });
      },
      onError: () => {
        toast({
          title: "Could not open the desktop",
          description: "The sandbox did not come up. Try again in a moment.",
          variant: "destructive",
        });
      },
    },
  });

  const isExpert = computer?.owner_kind === "expert";
  const hasDesktop = computer?.desktop != null;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
      <div className="flex items-center justify-between gap-3 rounded-lg border border-zinc-200 bg-white px-3 py-2">
        <div className="flex items-center gap-3 text-sm">
          <span className="flex items-center gap-1.5">
            <span className="text-zinc-500">Shell</span>
            <StatePill state={computer?.shell?.state ?? null} />
          </span>
          <span className="flex items-center gap-1.5">
            <span className="text-zinc-500">Desktop</span>
            <StatePill state={computer?.desktop?.state ?? null} />
          </span>
        </div>
        <Button
          variant="secondary"
          size="small"
          loading={isPending}
          disabled={computer != null && !computer.e2b_active}
          onClick={() => startDesktop({ sessionId })}
        >
          {stream ? "Refresh" : hasDesktop ? "Resume desktop" : "Start desktop"}
        </Button>
      </div>

      {stream ? (
        <div data-testid="session-desktop-stream">
          {desktopStreamRenderer.render({ kind: "desktop_stream", ...stream })}
        </div>
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-zinc-200 p-6 text-center">
          <Text variant="small-medium">No desktop on screen yet</Text>
          <Text variant="small" className="max-w-xs text-zinc-500">
            {isExpert
              ? "This chat runs on the expert's own computer. Start the desktop to watch it work, or ask for one in the chat."
              : "Start the desktop to watch this chat work in a real browser, or ask for one in the chat."}
          </Text>
        </div>
      )}

      {computer && !computer.e2b_active ? (
        <Text variant="small" className="text-amber-700">
          Cloud sandboxes are not configured on this deployment.
        </Text>
      ) : null}
    </div>
  );
}
