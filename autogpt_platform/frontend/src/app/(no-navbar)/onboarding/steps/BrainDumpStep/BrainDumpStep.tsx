"use client";

import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowReloadHorizontalIcon,
  Cancel01Icon,
  Loading03Icon,
  Mic01Icon,
  SentIcon,
} from "@hugeicons/core-free-icons";
import { type IconSvgElement } from "@hugeicons/react";
import { cn } from "@/lib/utils";
import { FailureState } from "./components/FailureState";
import { DEFAULT_GLASS_PARAMS } from "@/components/molecules/GlassOrb/GlassSurface";
import { ElapsedTime } from "./components/ElapsedTime";
import { MicButton, OrbScreen } from "./components/MicButton";
import { PrivacyNote } from "./components/PrivacyNote";
import { RecordingStatus } from "./components/RecordingStatus";
import { RecoveryPrompt } from "./components/RecoveryPrompt";
import { RevealGroup, RevealItem } from "@/components/atoms/Reveal/Reveal";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import { TypedFallback } from "./components/TypedFallback";
import { OrbSelector, OrbVariant } from "./components/OrbSelector";
import { DEFAULT_WAVY_ORB_SETTINGS } from "./components/WavyOrb";
import { ringProgress } from "./helpers";
import { ScreenState, useBrainDumpStep } from "./useBrainDumpStep";

const FAILURE_HEADLINE = "That didn't go through.";
const TIME_LIMIT_CAPTION =
  "That's 30 minutes — the most we record in one go. Saving all of it…";

export function BrainDumpStep() {
  const dump = useBrainDumpStep();
  const prefersReducedMotion = useReducedMotion();
  const [orbVariant, setOrbVariant] = useState<OrbVariant>("glass");
  const isRecording = dump.screen === "recording";
  const isProcessing = dump.screen === "processing";
  const isMicScreen = dump.screen === "rest" || isRecording;
  const isTyping = dump.screen === "typing";
  const showSubline = dump.screen !== "failed" && dump.screen !== "recovery";
  // rest → recording → processing all share one orb, so it is never
  // unmounted between them: only the glyph and the ring change.
  const orbScreen = toOrbScreen(dump.screen);

  function orbClick(screen: OrbScreen) {
    if (screen === "processing") return undefined;
    if (screen === "failed") return dump.handleRetry;
    return screen === "rest" ? dump.handleStart : undefined;
  }

  return (
    <>
      <AnimatePresence>
        {isRecording && (
          <motion.div
            className="fixed inset-0 z-40 bg-[#F6F7F8]/90 backdrop-blur-xl"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            aria-hidden
          />
        )}
      </AnimatePresence>

      <RevealGroup
        className={
          isRecording
            ? "fixed inset-0 z-50 flex w-full flex-col items-center justify-center px-4"
            : cn(
                "-mt-44 flex w-full flex-col items-center gap-12 px-4",
                isTyping ? "max-w-4xl" : "max-w-2xl",
              )
        }
      >
        {isRecording && (
          <div className="absolute left-1/2 top-8 -translate-x-1/2">
            <ElapsedTime seconds={dump.elapsedSeconds} />
          </div>
        )}

        <div
          className={cn(
            "absolute right-4 top-4 flex items-center gap-2 sm:right-6 sm:top-6 sm:gap-5",
            isRecording && "hidden",
          )}
        >
          {orbScreen && (
            <OrbSelector value={orbVariant} onChange={setOrbVariant} />
          )}
          {isRecording && (
            <Button
              variant="secondary"
              size="small"
              onClick={dump.handleRestart}
            >
              Restart
            </Button>
          )}
          {/* Skipping mid-submit would advance the wizard a second time
              behind the finalize that is already in flight, landing past
              the last step on a blank screen. */}
          {!isProcessing && (
            <button
              type="button"
              onClick={dump.handleSkip}
              className="text-sm text-zinc-400 transition-colors hover:text-zinc-700"
            >
              Skip for now
            </button>
          )}
        </div>

        <div
          className={cn(
            "mx-auto flex w-full max-w-2xl flex-col items-center gap-2 px-4 text-center",
            isRecording && "hidden",
          )}
        >
          <RevealItem>
            <Text variant="h4">
              {dump.screen === "failed" ? FAILURE_HEADLINE : dump.headline}
            </Text>
          </RevealItem>
          {showSubline && (
            <RevealItem>
              <Text
                variant="large"
                className="!text-zinc-500 md:whitespace-nowrap"
              >
                Just talk.{" "}
                <span className="bg-gradient-to-r from-purple-500 to-indigo-500 bg-clip-text text-transparent">
                  AutoPilot
                </span>{" "}
                listens, remembers, and starts taking work off your plate.
              </Text>
            </RevealItem>
          )}
        </div>

        {orbScreen && (
          <RevealItem blur={false} className="flex flex-col items-center gap-4">
            <motion.div
              className={cn(
                "will-change-transform",
                isRecording && orbVariant === "wavy" && "mb-20",
              )}
              animate={{
                scale: isRecording ? (prefersReducedMotion ? 1.12 : 1.3) : 1,
              }}
              transition={{
                duration: prefersReducedMotion ? 0.15 : 0.28,
                ease: [0.32, 0.72, 0, 1],
              }}
            >
              <MicButton
                screen={orbScreen}
                progress={ringProgress(dump.elapsedSeconds)}
                audioStream={dump.audioStream}
                glassParams={DEFAULT_GLASS_PARAMS}
                variant={orbVariant}
                wavySettings={DEFAULT_WAVY_ORB_SETTINGS}
              />
            </motion.div>
            {orbScreen === "recording" ? (
              <RecordingControls
                onStop={dump.handleStop}
                onSend={dump.handleDone}
                onRetry={dump.handleRestart}
                elapsedSeconds={dump.elapsedSeconds}
                showSilenceNudge={dump.showSilenceNudge}
                isOffline={dump.isOffline}
              />
            ) : orbScreen !== "processing" ? (
              <OrbControlButton
                screen={orbScreen}
                onClick={orbClick(orbScreen)}
              />
            ) : null}
            {/* Both slots keep their height across rest → recording →
                processing, so advancing a screen swaps their contents without
                nudging the orb or the headline. Failure has its own layout
                below the orb and needs neither. */}
            {orbScreen !== "failed" && !isRecording && (
              <>
                <div className="flex h-10 w-full items-center justify-center">
                  <SwapFade
                    swapKey={orbScreen}
                    className="flex w-full justify-center"
                  >
                    <OrbCaption
                      screen={orbScreen}
                      reachedTimeLimit={dump.reachedTimeLimit}
                    />
                  </SwapFade>
                </div>
                <div className="flex h-10 items-center justify-center" />
              </>
            )}
          </RevealItem>
        )}

        {dump.screen === "recovery" && dump.recoverable && (
          <RevealItem>
            <RecoveryPrompt
              durationSecs={dump.recoverable.durationSecs}
              onResume={dump.handleResumeRecovered}
              onDiscard={dump.handleDiscardRecovered}
            />
          </RevealItem>
        )}

        {dump.screen === "failed" && (
          <RevealItem>
            <FailureState
              onDownload={dump.handleDownloadRecording}
              onSkip={dump.handleSkip}
            />
          </RevealItem>
        )}

        {dump.screen === "typing" && (
          <RevealItem className="w-full">
            <TypedFallback
              value={dump.typedText}
              onChange={dump.setTypedText}
              onSubmit={dump.handleSubmitTyped}
            />
          </RevealItem>
        )}
      </RevealGroup>

      {/* Viewport-anchored, and kept outside the reveal group: an ancestor
          that animates `filter` or `transform` would turn these into
          absolutely positioned elements. */}
      {!isRecording &&
        (isMicScreen ||
          isTyping ||
          dump.screen === "failed" ||
          dump.screen === "recovery") && (
          <div className="fixed inset-x-0 bottom-32 flex justify-center px-4">
            <SwapFade swapKey={dump.screen}>
              {(dump.screen === "rest" || dump.screen === "failed") && (
                <Button
                  variant="ghost"
                  size="small"
                  onClick={dump.showTyping}
                  className="underline underline-offset-4"
                >
                  type instead
                </Button>
              )}
              {isTyping && !dump.isMicBlocked && (
                <Button
                  variant="ghost"
                  size="small"
                  onClick={dump.showRecording}
                  className="underline underline-offset-4"
                >
                  record instead
                </Button>
              )}
              {dump.screen === "recovery" && (
                <Button
                  variant="ghost"
                  size="small"
                  onClick={dump.handleTypeInsteadOfRecovered}
                  className="underline underline-offset-4"
                >
                  type instead
                </Button>
              )}
            </SwapFade>
          </div>
        )}

      {showSubline && !isRecording && <PrivacyNote />}
    </>
  );
}

function toOrbScreen(screen: ScreenState): OrbScreen | null {
  if (screen === "typing" || screen === "recovery") return null;
  return screen;
}

function OrbCaption({
  screen,
  reachedTimeLimit,
}: {
  screen: OrbScreen;
  reachedTimeLimit: boolean;
}) {
  if (screen === "processing") {
    return (
      <Text variant="lead" className="!text-base !text-zinc-500">
        {reachedTimeLimit ? TIME_LIMIT_CAPTION : "Got it. One second…"}
      </Text>
    );
  }

  return null;
}

function OrbControlButton({
  screen,
  onClick,
}: {
  screen: "rest" | "failed";
  onClick?: () => void;
}) {
  const ariaLabel = screen === "failed" ? "Try again" : "Start talking";

  return (
    <Button
      variant="icon"
      size="icon"
      onClick={onClick}
      aria-label={ariaLabel}
      className="mt-4 border border-black/5 bg-white shadow-sm hover:border-black/5 hover:bg-zinc-50"
    >
      <SwapFade swapKey={screen} className="flex items-center justify-center">
        {screen === "failed" ? (
          <Icon icon={ArrowReloadHorizontalIcon} size={22} strokeWidth={1.5} />
        ) : (
          <Icon icon={Mic01Icon} size={22} strokeWidth={1.5} />
        )}
      </SwapFade>
    </Button>
  );
}

function RecordingControls({
  onStop,
  onSend,
  onRetry,
  elapsedSeconds,
  showSilenceNudge,
  isOffline,
}: {
  onStop: () => Promise<void>;
  onSend: () => Promise<void>;
  onRetry: () => Promise<void>;
  elapsedSeconds: number;
  showSilenceNudge: boolean;
  isOffline: boolean;
}) {
  const [pendingAction, setPendingAction] = useState<RecordingAction | null>(
    null,
  );

  async function runAction(
    action: RecordingAction,
    callback: () => Promise<void>,
  ) {
    if (pendingAction) return;
    setPendingAction(action);
    try {
      await callback();
    } finally {
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

type RecordingAction = "cancel" | "send" | "retry";

function RecordingControlButton({
  label,
  pendingLabel,
  icon,
  action,
  pendingAction,
  onClick,
  primary = false,
}: {
  label: string;
  pendingLabel: string;
  icon: IconSvgElement;
  action: RecordingAction;
  pendingAction: RecordingAction | null;
  onClick: () => void;
  primary?: boolean;
}) {
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
