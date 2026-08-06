"use client";

import { useRef, useState } from "react";
import * as Sentry from "@sentry/nextjs";
import { Button } from "@/components/atoms/Button/Button";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import {
  ArrowCounterClockwiseIcon,
  PaperPlaneTiltIcon,
  SpinnerGapIcon,
  XIcon,
  type Icon as PhosphorIcon,
} from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { RecordingStatus } from "../RecordingStatus";

interface Props {
  onStop: () => Promise<void>;
  onSend: () => Promise<void>;
  onRetry: () => Promise<void>;
  elapsedSeconds: number;
  showSilenceNudge: boolean;
  isOffline: boolean;
}

type RecordingAction = "cancel" | "send" | "retry";

export function RecordingControls({
  onStop,
  onSend,
  onRetry,
  elapsedSeconds,
  showSilenceNudge,
  isOffline,
}: Props) {
  const [pendingAction, setPendingAction] = useState<RecordingAction | null>(
    null,
  );
  const isActionPendingRef = useRef(false);

  async function runAction(
    action: RecordingAction,
    callback: () => Promise<void>,
  ) {
    if (isActionPendingRef.current) return;
    isActionPendingRef.current = true;
    setPendingAction(action);
    try {
      await callback();
    } catch (error) {
      Sentry.captureException(error, {
        tags: { component: "RecordingControls", action },
      });
    } finally {
      isActionPendingRef.current = false;
      setPendingAction(null);
    }
  }

  const pendingStatus =
    pendingAction === "cancel"
      ? "Discarding this take…"
      : pendingAction === "send"
        ? "Sending your recording…"
        : pendingAction === "retry"
          ? "Starting a fresh take…"
          : null;

  return (
    <div className="mt-16 flex flex-col items-center gap-3">
      <div className="flex items-center gap-4">
        <RecordingControlButton
          label="Cancel recording"
          pendingLabel="Canceling recording"
          icon={XIcon}
          action="cancel"
          pendingAction={pendingAction}
          onClick={() => void runAction("cancel", onStop)}
        />
        <RecordingControlButton
          label="Send recording"
          pendingLabel="Sending recording"
          icon={PaperPlaneTiltIcon}
          action="send"
          pendingAction={pendingAction}
          onClick={() => void runAction("send", onSend)}
          primary
        />
        <RecordingControlButton
          label="Retry recording"
          pendingLabel="Restarting recording"
          icon={ArrowCounterClockwiseIcon}
          action="retry"
          pendingAction={pendingAction}
          onClick={() => void runAction("retry", onRetry)}
        />
      </div>
      <div
        data-testid="recording-feedback-slot"
        className="relative h-8 w-80 max-w-[calc(100vw-2rem)]"
      >
        <div aria-live="polite" className="absolute inset-x-0 top-0">
          {pendingStatus ? (
            <div className="flex h-8 items-start justify-center pt-2">
              <SwapFade swapKey={pendingAction ?? "idle"}>
                <p className="text-sm text-zinc-500">{pendingStatus}</p>
              </SwapFade>
            </div>
          ) : (
            <RecordingStatus
              elapsedSeconds={elapsedSeconds}
              showSilenceNudge={showSilenceNudge}
              isOffline={isOffline}
            />
          )}
        </div>
      </div>
    </div>
  );
}

interface RecordingControlButtonProps {
  label: string;
  pendingLabel: string;
  icon: PhosphorIcon;
  action: RecordingAction;
  pendingAction: RecordingAction | null;
  onClick: () => void;
  primary?: boolean;
}

function RecordingControlButton({
  label,
  pendingLabel,
  icon,
  action,
  pendingAction,
  onClick,
  primary = false,
}: RecordingControlButtonProps) {
  const isLoading = pendingAction === action;
  const isInactive = pendingAction !== null && !isLoading;
  const ControlIcon = icon;

  return (
    <Button
      variant="icon"
      size="icon"
      onClick={onClick}
      disabled={pendingAction !== null}
      aria-label={isLoading ? pendingLabel : label}
      aria-busy={isLoading}
      className={cn(
        "border shadow-sm transition-[transform,opacity,background-color] duration-150 ease-out active:scale-[0.97]",
        primary
          ? "border-black/10 bg-zinc-950 text-white hover:bg-zinc-800"
          : "border-black/5 bg-white hover:bg-zinc-50",
        isInactive && "opacity-40",
      )}
    >
      <span className="grid size-[22px] place-items-center [&>*]:[grid-area:1/1]">
        <SwapFade
          swapKey={isLoading ? "loading" : "idle"}
          mode="sync"
          className="flex size-[22px] items-center justify-center"
        >
          {isLoading ? (
            <SpinnerGapIcon
              size={22}
              weight="light"
              className="motion-safe:animate-spin"
              data-testid="recording-control-loader"
            />
          ) : (
            <ControlIcon size={22} weight="light" />
          )}
        </SwapFade>
      </span>
    </Button>
  );
}
