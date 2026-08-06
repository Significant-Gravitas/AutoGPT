"use client";

import { useRef, useState } from "react";
import * as Sentry from "@sentry/nextjs";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import {
  ArrowReloadHorizontalIcon,
  Cancel01Icon,
  Loading03Icon,
  SentIcon,
} from "@hugeicons/core-free-icons";
import { type IconSvgElement } from "@hugeicons/react";
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
          icon={Cancel01Icon}
          action="cancel"
          pendingAction={pendingAction}
          onClick={() => void runAction("cancel", onStop)}
        />
        <RecordingControlButton
          label="Send recording"
          pendingLabel="Sending recording"
          icon={SentIcon}
          action="send"
          pendingAction={pendingAction}
          onClick={() => void runAction("send", onSend)}
          primary
        />
        <RecordingControlButton
          label="Retry recording"
          pendingLabel="Restarting recording"
          icon={ArrowReloadHorizontalIcon}
          action="retry"
          pendingAction={pendingAction}
          onClick={() => void runAction("retry", onRetry)}
        />
      </div>
      <div
        data-testid="recording-feedback-slot"
        className="relative h-8 w-80 max-w-[calc(100vw-2rem)]"
      >
        <div className="absolute inset-x-0 top-0">
          {pendingStatus ? (
            <div
              aria-live="polite"
              className="flex h-8 items-start justify-center pt-2"
            >
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
  icon: IconSvgElement;
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
            <Icon
              icon={Loading03Icon}
              size={22}
              strokeWidth={1.5}
              className="motion-safe:animate-spin"
              data-testid="recording-control-loader"
            />
          ) : (
            <Icon icon={icon} size={22} strokeWidth={1.5} />
          )}
        </SwapFade>
      </span>
    </Button>
  );
}
