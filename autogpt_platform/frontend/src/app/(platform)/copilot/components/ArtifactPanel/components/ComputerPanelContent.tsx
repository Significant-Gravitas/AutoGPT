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

function Pill({ tone, label }: { tone: "on" | "off" | "none"; label: string }) {
  return (
    <span
      className={cn(
        "rounded-full px-2 py-0.5 text-xs font-medium",
        tone === "on"
          ? "bg-emerald-50 text-emerald-700"
          : tone === "off"
            ? "bg-zinc-100 text-zinc-600"
            : "bg-zinc-50 text-zinc-500",
      )}
    >
      {label}
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
  const box = computer?.box ?? null;
  const machineLabel = !box
    ? "None"
    : box.state === "running"
      ? "Running"
      : "Suspended";
  const machineTone = !box ? "none" : box.state === "running" ? "on" : "off";
  const screenOn = computer?.screen_on === true;
  const actionLabel = stream
    ? "Refresh"
    : screenOn
      ? "Open desktop"
      : "Turn on screen";

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
      <div className="flex items-center justify-between gap-3 rounded-lg border border-zinc-200 bg-white px-3 py-2">
        <div className="flex items-center gap-3 text-sm">
          <span className="flex items-center gap-1.5">
            <span className="text-zinc-500">Machine</span>
            <Pill tone={machineTone} label={machineLabel} />
          </span>
          <span className="flex items-center gap-1.5">
            <span className="text-zinc-500">Screen</span>
            <Pill
              tone={!box ? "none" : screenOn ? "on" : "off"}
              label={!box ? "None" : screenOn ? "On" : "Off"}
            />
          </span>
        </div>
        <Button
          variant="secondary"
          size="small"
          loading={isPending}
          disabled={computer != null && !computer.e2b_active}
          onClick={() => startDesktop({ sessionId })}
        >
          {actionLabel}
        </Button>
      </div>

      {stream ? (
        <div data-testid="session-desktop-stream">
          {desktopStreamRenderer.render({ kind: "desktop_stream", ...stream })}
        </div>
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-zinc-200 p-6 text-center">
          <Text variant="small-medium">Screen is not on this panel yet</Text>
          <Text variant="small" className="max-w-xs text-zinc-500">
            {isExpert
              ? "This chat runs on the expert's own computer. Turn on its screen to watch it work in a real browser, or ask for one in the chat."
              : "Turn on this chat's screen to watch it work in a real browser, or ask for one in the chat. It is the same machine your commands run in."}
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
