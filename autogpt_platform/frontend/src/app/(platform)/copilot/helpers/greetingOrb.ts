// Otto's avatar is one element across the greeting's arrival, not two:
// GreetingLoader renders it centered, OnboardingIntroCard renders it in
// the heading, and framer moves it between the two because they share
// this id. Both live here so neither component owns the other's values.
export const GREETING_ORB_LAYOUT_ID = "onboarding-greeting-orb";

export const ORB_FLIP_TRANSITION = {
  type: "spring",
  bounce: 0.15,
  duration: 0.55,
} as const;

// The same trip with prefers-reduced-motion on: the orb is where it
// belongs on the next frame instead of springing across the page.
export const ORB_FLIP_TRANSITION_REDUCED = { duration: 0 } as const;

// The purple the orb's blobs blend into — the name mirrors it.
export const ORB_PURPLE = "#8a4dff";
