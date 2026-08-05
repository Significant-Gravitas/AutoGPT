"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  ArrowReloadHorizontalIcon,
  HandIcon,
  Mic01Icon,
} from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { AudioWaveform } from "@/app/(platform)/copilot/components/ChatInput/components/AudioWaveform";
import { GlassParams } from "@/components/molecules/GlassOrb/GlassSurface";
import { OrbFrame } from "./OrbFrame";

export type OrbScreen = "rest" | "recording" | "processing" | "failed";

const GLYPH_CLASS =
  "text-white/95 drop-shadow-[0_2px_12px_rgba(90,40,180,0.5)]";

const ARIA_LABEL: Record<OrbScreen, string | undefined> = {
  rest: "Start recording",
  recording: "I'm done",
  processing: undefined,
  failed: "Try again",
};

interface Props {
  screen: OrbScreen;
  progress: number;
  audioStream: MediaStream | null;
  glassParams: GlassParams;
  onClick?: () => void;
}

export function MicButton({
  screen,
  progress,
  audioStream,
  glassParams,
  onClick,
}: Props) {
  const prefersReducedMotion = useReducedMotion();

  return (
    <OrbFrame
      glassParams={glassParams}
      progress={screen === "recording" ? progress : undefined}
      isLoading={screen === "processing"}
      onClick={onClick}
      ariaLabel={ARIA_LABEL[screen]}
    >
      {/* Only the glyph inside the orb changes as the step advances — the
          orb, the ring and the button itself stay mounted throughout. */}
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={screen}
          initial={
            prefersReducedMotion
              ? { opacity: 0 }
              : { opacity: 0, scale: 0.72, filter: "blur(6px)" }
          }
          animate={
            prefersReducedMotion
              ? { opacity: 1 }
              : { opacity: 1, scale: 1, filter: "blur(0px)" }
          }
          exit={
            prefersReducedMotion
              ? { opacity: 0 }
              : { opacity: 0, scale: 0.72, filter: "blur(6px)" }
          }
          transition={{ duration: 0.2, ease: [0.32, 0.72, 0, 1] }}
          className="flex items-center justify-center"
        >
          <OrbGlyph screen={screen} audioStream={audioStream} />
        </motion.span>
      </AnimatePresence>
    </OrbFrame>
  );
}

function OrbGlyph({
  screen,
  audioStream,
}: {
  screen: OrbScreen;
  audioStream: MediaStream | null;
}) {
  if (screen === "processing") {
    return <Icon icon={HandIcon} size={56} className={GLYPH_CLASS} />;
  }

  if (screen === "failed") {
    return (
      <Icon
        icon={ArrowReloadHorizontalIcon}
        size={56}
        className={GLYPH_CLASS}
      />
    );
  }

  if (screen === "recording") {
    return (
      <AudioWaveform
        stream={audioStream}
        barCount={9}
        barWidth={4}
        barGap={4}
        barColor="rgba(255,255,255,0.95)"
        minBarHeight={7}
        maxBarHeight={44}
      />
    );
  }

  return <Icon icon={Mic01Icon} size={56} className={GLYPH_CLASS} />;
}
