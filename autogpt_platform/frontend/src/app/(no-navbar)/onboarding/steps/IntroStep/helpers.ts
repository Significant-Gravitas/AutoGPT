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
