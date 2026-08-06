"use client";

import { GlassParams } from "@/components/molecules/GlassOrb/GlassSurface";
import { OrbVariant } from "./OrbSelector";
import { OrbVisual } from "./OrbVisual";
import { useAudioLevel } from "./useAudioLevel";
import { type WavyOrbSettings } from "./WavyOrb";

export const ORB_SIZE = 184;
const STROKE = 6;
const RADIUS = (ORB_SIZE - STROKE) / 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

// A quarter of the ring, so the gap reads as motion rather than a dotted line.
const LOADER_ARC = CIRCUMFERENCE * 0.25;

interface Props {
  glassParams: GlassParams;
  variant: OrbVariant;
  audioStream: MediaStream | null;
  wavySettings: WavyOrbSettings;
  // Omitted when there is nothing to meter — the arc is not rendered at all.
  progress?: number;
  // Indeterminate: a single arc chasing the ring while work is in flight.
  isLoading?: boolean;
}

// The orb as it appears everywhere in this step: the same glass ball seated
// in the same neumorphic ring, whether or not it is interactive.
export function OrbFrame({
  glassParams,
  variant,
  audioStream,
  wavySettings,
  progress,
  isLoading,
}: Props) {
  const isRecording = progress !== undefined;
  const audioLevel = useAudioLevel(
    isRecording && variant !== "wavy" ? audioStream : null,
  );

  return (
    <div
      data-testid="orb-frame"
      className="relative"
      style={{ width: ORB_SIZE, height: ORB_SIZE }}
    >
      {variant === "glass" && (
        <div
          data-testid="orb-decorative-ring"
          className="pointer-events-none absolute rounded-full bg-[#f1f1f4]"
          style={{
            inset: -glassParams.ringWidth,
            boxShadow: `${glassParams.ringDepth}px ${glassParams.ringDepth}px ${glassParams.ringDepth * 2.3}px rgba(166,171,189,${glassParams.ringDark}), -${glassParams.ringDepth}px -${glassParams.ringDepth}px ${glassParams.ringDepth * 2.3}px #ffffff, inset 1px 1px 3px rgba(255,255,255,0.95), inset -1px -1px 3px rgba(166,171,189,${glassParams.ringDark * 0.55})`,
          }}
        />
      )}
      {/* A depth meter, not a limit — the ring fills toward three minutes
          and then simply holds, so passing it reads as an achievement
          rather than a warning. */}
      {variant === "glass" && progress !== undefined && (
        <svg
          data-testid="orb-progress-ring"
          className="pointer-events-none absolute inset-0 -rotate-90"
          width={ORB_SIZE}
          height={ORB_SIZE}
          aria-hidden
        >
          <circle
            cx={ORB_SIZE / 2}
            cy={ORB_SIZE / 2}
            r={RADIUS}
            fill="none"
            stroke="#C084FC"
            strokeOpacity={0.95}
            strokeWidth={STROKE}
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={CIRCUMFERENCE * (1 - progress)}
            className="transition-[stroke-dashoffset] duration-500 ease-linear [filter:drop-shadow(0_0_6px_rgba(192,132,252,0.6))]"
          />
        </svg>
      )}
      {variant === "glass" && isLoading && (
        <svg
          className="absolute inset-0 motion-safe:animate-spin"
          width={ORB_SIZE}
          height={ORB_SIZE}
          aria-hidden
        >
          <circle
            cx={ORB_SIZE / 2}
            cy={ORB_SIZE / 2}
            r={RADIUS}
            fill="none"
            stroke="#C084FC"
            strokeOpacity={0.95}
            strokeWidth={STROKE}
            strokeLinecap="round"
            strokeDasharray={`${LOADER_ARC} ${CIRCUMFERENCE - LOADER_ARC}`}
          />
        </svg>
      )}

      <div
        className={
          variant === "glass"
            ? "absolute inset-[8px] rounded-full"
            : "absolute inset-0 rounded-full"
        }
      >
        <OrbVisual
          variant={variant}
          glassParams={glassParams}
          audioStream={audioStream}
          audioLevel={audioLevel}
          wavySettings={wavySettings}
          isRecording={isRecording}
          isLoading={isLoading ?? false}
        />
      </div>
    </div>
  );
}
