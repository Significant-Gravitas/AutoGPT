"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { PauseIcon, PlayIcon, RefreshIcon } from "@hugeicons/core-free-icons";
import { useId } from "react";
import { useWorkflowsMovedWalkthrough } from "./useWorkflowsMovedWalkthrough";

type Player = ReturnType<typeof useWorkflowsMovedWalkthrough>;

export function WorkflowsMovedWalkthrough() {
  const player = useWorkflowsMovedWalkthrough();
  const descriptionID = useId();

  return (
    <div
      ref={player.containerRef}
      className="overflow-hidden rounded-2xl border border-border-base bg-surface-muted"
    >
      <p id={descriptionID} className="sr-only">
        In the sidebar, open Team. Select Otto, then select the Workflows tab to
        find your existing workflows. Select a workflow name to see its details.
        Choose Setup your task to review the inputs and start or schedule a run.
        This video has no sound.
      </p>
      {player.playback === "failed" ? (
        <div className="flex aspect-video items-center justify-center px-8">
          <p
            ref={player.fallbackRef}
            role="status"
            tabIndex={-1}
            className="max-w-sm text-center text-sm leading-relaxed text-foreground-secondary outline-none"
          >
            The walkthrough couldn&apos;t load. Open Team, select Otto, then
            choose Workflows.
          </p>
        </div>
      ) : (
        <>
          <div className="relative">
            <video
              ref={player.videoRef}
              src="/videos/workflows-moved.mp4"
              poster="/videos/workflows-moved-poster.webp"
              aria-label="How to find your workflows"
              aria-describedby={descriptionID}
              className="block aspect-video w-full bg-surface-hover object-contain"
              muted
              playsInline
              preload="metadata"
              controls={player.hasStarted}
              onPlay={player.handlePlay}
              onPause={player.handlePause}
              onEnded={player.handleEnded}
              onError={player.handleError}
              onLoadedMetadata={player.handleMetadata}
              onDurationChange={player.handleMetadata}
            />
            <WalkthroughControl player={player} />
          </div>
          {player.playback === "blocked" && (
            <p
              role="status"
              className="border-t border-border-base px-4 py-3 text-xs leading-relaxed text-foreground-secondary"
            >
              The video couldn&apos;t start. Try playing it again, or open Team,
              select Otto, then choose Workflows.
            </p>
          )}
        </>
      )}
    </div>
  );
}

function WalkthroughControl({ player }: { player: Player }) {
  const isPlaying = player.playback === "playing";
  const hasEnded = player.playback === "ended";
  const action = isPlaying ? "Pause" : hasEnded ? "Replay" : "Play";
  const icon = isPlaying ? PauseIcon : hasEnded ? RefreshIcon : PlayIcon;

  return (
    <>
      {!player.hasStarted && (
        <div
          aria-hidden
          // TODO(tokens): no fitting role for bg-zinc-950/15 #09090B; #09090B is foreground, a text colour
          className="pointer-events-none absolute inset-x-0 bottom-14 top-0 bg-zinc-950/15"
        />
      )}
      <div className="flex min-h-14 items-center border-t border-border-base/80 bg-surface px-4 py-2">
        <div className={player.hasStarted ? "pr-24" : undefined}>
          <p className="text-sm font-medium text-foreground-body">
            See where they live
          </p>
          <p className="mt-0.5 text-xs text-foreground-tertiary">
            {player.durationLabel && (
              <>
                <span aria-label="Video duration">{player.durationLabel}</span>
                <span aria-hidden> · </span>
              </>
            )}
            No sound needed
          </p>
        </div>
      </div>
      <Button
        type="button"
        variant={player.hasStarted ? "secondary" : "primary"}
        size={player.hasStarted ? "small" : "icon"}
        withTooltip={false}
        aria-label={`${action} walkthrough`}
        aria-busy={player.isStarting}
        onClick={player.togglePlayback}
        className={
          player.hasStarted
            ? "absolute bottom-2.5 right-3"
            : // TODO(tokens): no exact role for bg-violet-600 (#7C3AED), focus-visible:ring-violet-600 (#7C3AED); no fitting role for hover:bg-violet-700 #6D28D9; #6D28D9 is accent-strong, a text colour
              "absolute left-1/2 top-[calc(50%-1.75rem)] size-16 -translate-x-1/2 -translate-y-1/2 rounded-full border-white/20 bg-violet-600 text-action-foreground shadow-xl hover:border-white/30 hover:bg-violet-700 focus-visible:ring-2 focus-visible:ring-violet-600 focus-visible:ring-offset-4"
        }
      >
        <Icon icon={icon} size={player.hasStarted ? 14 : 26} aria-hidden />
        <span className={player.hasStarted ? undefined : "sr-only"}>
          {action}
        </span>
      </Button>
    </>
  );
}
