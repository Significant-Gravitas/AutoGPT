"use client";

import { useEffect, useRef, useState } from "react";
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
import type { IconSvgElement } from "@hugeicons/react";
import { cn } from "@/lib/utils";
import { RecordingStatus } from "../RecordingStatus";
import { CancelRecordingDialog } from "./CancelRecordingDialog";

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
  const [isCancelDialogOpen, setIsCancelDialogOpen] = useState(false);
  const isActionPendingRef = useRef(false);
  const cancelControlRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (pendingAction !== "cancel") return;
    cancelControlRef.current?.querySelector("button")?.focus();
  }, [pendingAction]);

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
        <div ref={cancelControlRef}>
          <RecordingControlButton
            label="Cancel recording"
            pendingLabel="Canceling recording"
            icon={Cancel01Icon}
            action="cancel"
            pendingAction={pendingAction}
            onClick={() => setIsCancelDialogOpen(true)}
          />
        </div>
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
        <div aria-live="polite" className="absolute inset-x-0 top-0">
          {pendingStatus && (
            <div className="flex h-8 items-start justify-center pt-2">
              <SwapFade swapKey={pendingAction ?? "idle"}>
                <p className="text-sm text-zinc-500">{pendingStatus}</p>
              </SwapFade>
            </div>
          )}
        </div>
        {!pendingStatus && (
          <RecordingStatus
            elapsedSeconds={elapsedSeconds}
            showSilenceNudge={showSilenceNudge}
            isOffline={isOffline}
          />
        )}
      </div>
      <CancelRecordingDialog
        isOpen={isCancelDialogOpen}
        onOpenChange={setIsCancelDialogOpen}
        onConfirm={() => {
          setIsCancelDialogOpen(false);
          void runAction("cancel", onStop);
        }}
      />
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

  function handleClick() {
    if (pendingAction !== null) return;
    onClick();
  }

  return (
    <Button
      variant="icon"
      size="icon"
      onClick={handleClick}
      aria-disabled={pendingAction !== null}
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
              className="motion-safe:animate-spin"
              data-testid="recording-control-loader"
            />
          ) : (
            <Icon icon={icon} size={22} />
          )}
        </SwapFade>
      </span>
    </Button>
  );
}
