"use client";

import { ACTION_BUTTON_CLASS } from "@/app/(platform)/team/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { desktopStreamRenderer } from "@/components/contextual/OutputRenderers/renderers/DesktopStreamRenderer";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";
import {
  ComputerTerminalIcon,
  CpuIcon,
  Folder01Icon,
  SecurityIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { ReactNode } from "react";
import {
  describeMount,
  desktopActionLabel,
  formatResources,
  formatSandboxState,
  formatScreen,
} from "./helpers";
import { useExpertComputerSection } from "./useExpertComputerSection";

interface Props {
  expertId: string;
  expertName: string;
}

export function ExpertComputerSection({ expertId, expertName }: Props) {
  const { computer, isLoading, refetch, stream, openDesktop, isOpening } =
    useExpertComputerSection({ expertId });

  if (isLoading) {
    return (
      <section className="space-y-3">
        <Skeleton className="h-8 w-64 rounded-lg" />
        <Skeleton className="h-28 w-full rounded-lg" />
      </section>
    );
  }

  // A failed background poll must not tear down a live desktop the user is
  // watching; last-good data stays up and the next poll refreshes it.
  if (!computer) {
    return (
      <ErrorCard
        context="this expert's computer"
        hint="We could not load the sandbox status."
        onRetry={() => refetch()}
      />
    );
  }

  const box = computer.box;
  const running = box?.state === "running";
  const resources = formatResources(box);
  const mounts = Object.entries(computer.mounts ?? {});

  const status = running ? "Running" : box ? "Suspended" : "None";

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <Text variant="large-medium">{`${expertName}'s computer`}</Text>
          <Text variant="small" className="mt-1 max-w-prose !text-zinc-500">
            One persistent cloud machine only {expertName} uses: the shell it
            runs commands in and, when needed, its screen. It is suspended
            between turns and costs nothing while suspended; installed tools and
            files stay put.
          </Text>
        </div>
      </div>

      {!computer.e2b_active ? (
        <div className="rounded-xl bg-amber-50 px-4 py-2.5 ring-1 ring-inset ring-amber-200">
          <Text variant="small" className="text-amber-700">
            Cloud sandboxes are not configured on this deployment.
          </Text>
        </div>
      ) : null}

      {/* The screen itself: the live desktop when there is one, otherwise a
          dark monitor showing the machine's state and the way in. */}
      <div className="overflow-hidden rounded-2xl bg-white p-2 smooth-shadow-ring-sm">
        {stream ? (
          <div className="flex flex-col gap-2">
            <div
              data-testid="expert-desktop-stream"
              className="overflow-hidden rounded-xl"
            >
              {desktopStreamRenderer.render(stream)}
            </div>
            {/* A finished turn suspends the machine and ends this stream, so
                the way back in stays next to it. */}
            <div className="flex justify-end px-1 pb-1">
              <Button
                variant="secondary"
                size="small"
                loading={isOpening}
                disabled={!computer.e2b_active}
                onClick={openDesktop}
              >
                Reconnect desktop
              </Button>
            </div>
          </div>
        ) : (
          <div className="relative flex min-h-[240px] flex-col items-center justify-center gap-4 overflow-hidden rounded-xl bg-gradient-to-br from-zinc-800 via-zinc-900 to-black px-6 py-8 text-center sm:aspect-[16/7] sm:min-h-0 sm:py-0">
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(255,255,255,0.08),transparent_60%)]"
            />
            <span
              className={cn(
                "relative flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
                running
                  ? "bg-emerald-500/15 text-emerald-300"
                  : "bg-white/10 text-zinc-300",
              )}
            >
              <span
                aria-hidden
                className={cn(
                  "size-1.5 rounded-full",
                  running ? "bg-emerald-400" : "bg-zinc-400",
                )}
              />
              {status}
            </span>
            <Text variant="body" className="relative max-w-md !text-zinc-300">
              {formatScreen(computer)}
            </Text>
            <Button
              variant="secondary"
              size="small"
              className={cn("relative", ACTION_BUTTON_CLASS)}
              loading={isOpening}
              disabled={!computer.e2b_active}
              onClick={openDesktop}
            >
              {desktopActionLabel(computer)}
            </Button>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <DetailTile icon={ComputerTerminalIcon} label="Machine">
          {box
            ? formatSandboxState(box)
            : "Created on the first message that runs a command."}
        </DetailTile>
        <DetailTile icon={CpuIcon} label="Resources">
          {resources
            ? `${resources}${box?.mounts_attached === false ? " · no volumes attached" : ""}`
            : "Assigned when the machine is created."}
        </DetailTile>
      </div>

      {mounts.length > 0 ? (
        <div className="rounded-2xl bg-white p-4 smooth-shadow-ring-sm">
          <Text variant="small-medium" className="!text-zinc-700">
            Volumes
          </Text>
          <ul className="mt-2 divide-y divide-zinc-100">
            {mounts.map(([path, name]) => (
              <li key={path} className="flex items-start gap-3 py-2.5">
                <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-zinc-200 bg-zinc-50 text-zinc-500">
                  <Icon icon={Folder01Icon} size={15} aria-hidden />
                </span>
                <span className="flex min-w-0 flex-col gap-0.5">
                  <code className="text-xs text-zinc-800">{path}</code>
                  <Text variant="small" className="!text-zinc-500">
                    {describeMount(path, computer)}{" "}
                    <span className="font-mono text-xs text-zinc-400">
                      {name}
                    </span>
                  </Text>
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="flex items-start gap-2.5 rounded-xl bg-zinc-100/80 px-4 py-3">
        <Icon
          icon={SecurityIcon}
          size={16}
          className="mt-0.5 shrink-0 text-zinc-500"
        />
        <Text variant="small" className="!text-zinc-600">
          {expertName} has full access to this machine, so anything signed in on
          its desktop is visible to the AI. Do not sign into personal accounts
          there.
        </Text>
      </div>
    </section>
  );
}

interface DetailTileProps {
  icon: IconSvgElement;
  label: string;
  children: ReactNode;
}

function DetailTile({ icon, label, children }: DetailTileProps) {
  return (
    <div className="flex items-start gap-3 rounded-2xl bg-white p-4 smooth-shadow-ring-sm">
      <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-zinc-200 bg-zinc-50 text-zinc-500">
        <Icon icon={icon} size={15} aria-hidden />
      </span>
      <span className="flex min-w-0 flex-col gap-0.5">
        <Text variant="small-medium" className="!text-zinc-700">
          {label}
        </Text>
        <Text variant="small" className="!text-zinc-500">
          {children}
        </Text>
      </span>
    </div>
  );
}
