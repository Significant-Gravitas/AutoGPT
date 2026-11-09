export type IntroSlideId = "team" | "autopilot";

export interface IntroSlide {
  eyebrow?: string;
  title: string;
  body?: string;
  cta: string;
}

export const INTRO_SLIDES: Record<IntroSlideId, IntroSlide> = {
  team: {
    eyebrow: "Welcome to AutoGPT",
    title: "Your new team, ready on day one",
    body: "Maria drafts your blog posts. Max finds your next customers.",
    cta: "Meet Otto",
  },
  autopilot: { title: "Meet Otto, your Head of AI.", cta: "Next" },
};
