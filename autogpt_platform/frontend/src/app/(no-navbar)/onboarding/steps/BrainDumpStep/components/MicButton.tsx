import { GlassParams } from "@/components/molecules/GlassOrb/GlassSurface";
import { OrbFrame } from "./OrbFrame";

export type OrbScreen = "rest" | "recording" | "processing" | "failed";

interface Props {
  screen: OrbScreen;
  progress: number;
  audioStream: MediaStream | null;
  glassParams: GlassParams;
}

export function MicButton({
  screen,
  progress,
  audioStream,
  glassParams,
}: Props) {
  return (
    <OrbFrame
      glassParams={glassParams}
      audioStream={audioStream}
      progress={screen === "recording" ? progress : undefined}
      isLoading={screen === "processing"}
    />
  );
}
