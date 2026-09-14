import type { Pose } from "@/components/molecules/BotAvatar/projection";

export type IntroSlideId = "team" | "autopilot";

export interface IntroSlide {
  title: string;
  body?: string;
}

export const INTRO_SLIDES: Record<IntroSlideId, IntroSlide> = {
  team: {
    title: "Your own team of AI experts.",
    body: "Go beyond chat. Get experts that research, create, and handle work for you.",
  },
  autopilot: { title: "Meet Otto, your Head of AI." },
};

export function facing(yawDeg: number, pitchDeg = 0): Partial<Pose> {
  return {
    yaw: (yawDeg * Math.PI) / 180,
    pitch: (pitchDeg * Math.PI) / 180,
  };
}
