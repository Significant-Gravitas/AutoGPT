import { GlassParams } from "@/components/molecules/GlassOrb/GlassSurface";
import { OrbFrame } from "./OrbFrame";
import { OrbVariant } from "./OrbSelector";
import { type WavyOrbSettings } from "./WavyOrb/helpers";

export type OrbScreen = "rest" | "recording" | "processing" | "failed";

interface Props {
  screen: OrbScreen;
  progress: number;
  audioStream: MediaStream | null;
  glassParams: GlassParams;
  variant: OrbVariant;
  wavySettings: WavyOrbSettings;
}

export function MicButton({
  screen,
  progress,
  audioStream,
  glassParams,
  variant,
  wavySettings,
}: Props) {
  return (
    <OrbFrame
      glassParams={glassParams}
      variant={variant}
      audioStream={audioStream}
      wavySettings={wavySettings}
      progress={screen === "recording" ? progress : undefined}
      isLoading={screen === "processing"}
    />
  );
}
