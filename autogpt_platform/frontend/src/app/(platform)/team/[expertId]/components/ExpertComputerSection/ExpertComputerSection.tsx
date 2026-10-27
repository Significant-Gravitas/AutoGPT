"use client";

import { ACTION_BUTTON_CLASS } from "@/app/(platform)/team/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  DesktopStreamPreview,
  isDesktopStream,
} from "@/components/contextual/OutputRenderers/renderers/DesktopStreamRenderer";
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
        textSize="body-md"
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
          <Text
            variant="small"
            className="mt-1 max-w-prose !text-foreground-tertiary"
          >
            One persistent cloud machine only {expertName} uses: the shell it
            runs commands in and, when needed, its screen. It is suspended
            between turns and costs nothing while suspended; installed tools and
            files stay put.
          </Text>
        </div>
      </div>

      {!computer.e2b_active ? (
        <div className="rounded-xl bg-warning-surface px-4 py-2.5 ring-1 ring-inset ring-warning-border">
          <Text variant="small" className="text-warning-foreground">
            Cloud sandboxes are not configured on this deployment.
          </Text>
        </div>
      ) : null}

      {/* The screen itself: the live desktop when there is one, otherwise a
          dark monitor showing the machine's state and the way in. */}
      <div className="overflow-hidden rounded-2xl bg-surface p-2 smooth-shadow-ring-sm">
        {stream ? (
          <div className="flex flex-col gap-2">
            <div
              data-testid="expert-desktop-stream"
              className="overflow-hidden rounded-xl"
            >
              {isDesktopStream(stream) ? (
                <DesktopStreamPreview value={stream} textSize="body-md" />
              ) : null}
            </div>
            {/* A finished turn suspends the machine and ends this stream, so
                the way back in stays next to it. */}
            <div className="flex justify-end px-1 pb-1">
              <Button
                variant="secondary"
                size="small"
                textSize="body-md"
                loading={isOpening}
                disabled={!computer.e2b_active}
                onClick={openDesktop}
              >
                Reconnect desktop
              </Button>
            </div>
          </div>
        ) : (
          <div
            // TODO(tokens): no fitting role for to-black #141414; #141414 is foreground-base (a text colour)
            className="relative flex min-h-[240px] flex-col items-center justify-center gap-4 overflow-hidden rounded-xl bg-gradient-to-br from-action via-action-hover to-black px-6 py-8 text-center sm:aspect-[16/7] sm:min-h-0 sm:py-0"
          >
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgb(var(--color-card)/0.08),transparent_60%)]"
            />
            <span
              className={cn(
                "relative flex items-center gap-1.5 rounded-full px-2.5 py-1",
                running
                  ? // TODO(tokens): no exact role: bg-emerald-500/15 #10B981
                    "bg-emerald-500/15"
                  : "bg-surface/10",
              )}
            >
              <span
                aria-hidden
                className={cn(
                  "size-1.5 rounded-full",
                  // TODO(tokens): no exact role: bg-emerald-400 #34D399; no fitting role for bg-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
                  running ? "bg-emerald-400" : "bg-zinc-400",
                )}
              />
              <Text
                variant="small-medium-tight"
                as="span"
                tone={running ? "success-light" : "disabled"}
              >
                {status}
              </Text>
            </span>
            <Text
              variant="body-md"
              className="relative max-w-md !text-foreground-disabled"
            >
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
        <div className="rounded-2xl bg-surface p-4 smooth-shadow-ring-sm">
          <Text variant="small-medium" className="!text-foreground-body">
            Volumes
          </Text>
          <ul className="mt-2 divide-y divide-separator">
            {mounts.map(([path, name]) => (
              <li key={path} className="flex items-start gap-3 py-2.5">
                <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border-base bg-surface-muted text-foreground-tertiary">
                  <Icon icon={Folder01Icon} size={15} aria-hidden />
                </span>
                <span className="flex min-w-0 flex-col gap-0.5">
                  <Text variant="small-mono-tight" as="code" tone="emphasis">
                    {path}
                  </Text>
                  <Text variant="small" className="!text-foreground-tertiary">
                    {describeMount(path, computer)}{" "}
                    <Text variant="small-mono-tight" as="span" tone="subtle">
                      {name}
                    </Text>
                  </Text>
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="flex items-start gap-2.5 rounded-xl bg-surface-hover/80 px-4 py-3">
        <Icon
          icon={SecurityIcon}
          size={16}
          className="mt-0.5 shrink-0 text-foreground-tertiary"
        />
        <Text variant="small" className="!text-foreground-secondary">
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
    <div className="flex items-start gap-3 rounded-2xl bg-surface p-4 smooth-shadow-ring-sm">
      <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border-base bg-surface-muted text-foreground-tertiary">
        <Icon icon={icon} size={15} aria-hidden />
      </span>
      <span className="flex min-w-0 flex-col gap-0.5">
        <Text variant="small-medium" className="!text-foreground-body">
          {label}
        </Text>
        <Text variant="small" className="!text-foreground-tertiary">
          {children}
        </Text>
      </span>
    </div>
  );
}
