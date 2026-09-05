"use client";

import type { SandboxSummary } from "@/app/api/__generated__/models/sandboxSummary";
import { ACTION_BUTTON_CLASS } from "@/app/(platform)/team/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { desktopStreamRenderer } from "@/components/contextual/OutputRenderers/renderers/DesktopStreamRenderer";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";
import { ComputerIcon, ComputerTerminalIcon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import {
  describeMount,
  desktopActionLabel,
  formatResources,
  formatSandboxState,
} from "./helpers";
import { useExpertComputerSection } from "./useExpertComputerSection";

interface Props {
  expertId: string;
  expertName: string;
  enabled: boolean;
}

function SandboxCard({
  title,
  icon,
  summary,
  idleHint,
}: {
  title: string;
  icon: IconSvgElement;
  summary: SandboxSummary | null | undefined;
  idleHint: string;
}) {
  const running = summary?.state === "running";
  const resources = formatResources(summary);
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Icon icon={icon} size={16} className="text-zinc-500" />
          <Text variant="small-medium">{title}</Text>
        </div>
        <span
          className={cn(
            "rounded-full px-2 py-0.5 text-xs font-medium",
            running
              ? "bg-emerald-50 text-emerald-700"
              : summary
                ? "bg-zinc-100 text-zinc-600"
                : "bg-zinc-50 text-zinc-500",
          )}
        >
          {running ? "Running" : summary ? "Suspended" : "None"}
        </span>
      </div>
      <Text variant="small" className="mt-2 text-zinc-600">
        {summary ? formatSandboxState(summary) : idleHint}
      </Text>
      {resources ? (
        <Text variant="small" className="mt-1 text-zinc-500">
          {resources}
          {summary?.mounts_attached === false ? " · no volumes attached" : ""}
        </Text>
      ) : null}
    </div>
  );
}

export function ExpertComputerSection({
  expertId,
  expertName,
  enabled,
}: Props) {
  const {
    computer,
    isLoading,
    isError,
    refetch,
    stream,
    openDesktop,
    isOpening,
  } = useExpertComputerSection({ expertId, enabled });

  if (isLoading) {
    return (
      <section className="space-y-3">
        <Skeleton className="h-8 w-64 rounded-lg" />
        <Skeleton className="h-28 w-full rounded-lg" />
      </section>
    );
  }

  if (isError || !computer) {
    return (
      <ErrorCard
        context="this expert's computer"
        hint="We could not load the sandbox status."
        onRetry={() => refetch()}
      />
    );
  }

  const mounts = Object.entries(computer.mounts ?? {});

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <Text variant="large-medium">{`${expertName}'s computer`}</Text>
          <Text variant="small" className="mt-1 max-w-prose text-zinc-500">
            A persistent cloud machine only {expertName} uses. It is suspended
            when idle and costs nothing while suspended; installed tools, logins
            and files stay put between chats.
          </Text>
        </div>
        <Button
          variant="primary"
          size="small"
          className={ACTION_BUTTON_CLASS}
          loading={isOpening}
          disabled={!computer.e2b_active}
          onClick={openDesktop}
        >
          {desktopActionLabel(computer)}
        </Button>
      </div>

      {!computer.e2b_active ? (
        <div className="rounded-lg bg-amber-50 px-4 py-2.5 ring-1 ring-inset ring-amber-200">
          <Text variant="small" className="text-amber-700">
            Cloud sandboxes are not configured on this deployment.
          </Text>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <SandboxCard
          title="Shell"
          icon={ComputerTerminalIcon}
          summary={computer.shell}
          idleHint="Created on the first message that runs a command."
        />
        <SandboxCard
          title="Desktop"
          icon={ComputerIcon}
          summary={computer.desktop}
          idleHint="Created the first time a task needs a browser or GUI app."
        />
      </div>

      {mounts.length > 0 ? (
        <div className="rounded-lg border border-zinc-200 bg-white p-4">
          <Text variant="small-medium">Volumes</Text>
          <ul className="mt-2 space-y-2">
            {mounts.map(([path, name]) => (
              <li key={path} className="flex flex-col gap-0.5">
                <code className="text-xs text-zinc-800">{path}</code>
                <Text variant="small" className="text-zinc-500">
                  {describeMount(path, computer)}{" "}
                  <span className="font-mono text-xs text-zinc-400">
                    {name}
                  </span>
                </Text>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {stream ? (
        <div data-testid="expert-desktop-stream">
          {desktopStreamRenderer.render(stream)}
        </div>
      ) : null}
    </section>
  );
}
