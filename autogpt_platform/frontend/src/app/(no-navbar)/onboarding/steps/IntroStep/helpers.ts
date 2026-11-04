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
    body: "Maria drafts your SEO articles. Max finds your next customers. Mina keeps the books. Otto runs the team, so you don't have to.",
    cta: "Meet Otto",
  },
  autopilot: { title: "Meet Otto, your Head of AI.", cta: "Next" },
};
