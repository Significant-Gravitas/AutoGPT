export const TOUR_GITHUB_URL =
  "https://github.com/Significant-Gravitas/AutoGPT";

/** The tour's marketing claim, matched to the page title ("…in 60 seconds").
 * The scripted animation runs shorter, but with reading and Enter presses a
 * real watch-through lands around here. */
export const TOUR_DEMO_CLAIM_SECONDS = 60;
/** Watch-time pitch for the next scenario on the idle nudge chip. */
export const TOUR_NEXT_SCENARIO_SECONDS = 45;

/** Outbound pricing links carry UTM params so DataFast can attribute
 * upgrades back to the exact tour placement that drove them. */
export function buildTourPricingUrl(placement: "end_card" | "sidebar_card") {
  return `https://agpt.co/pricing?utm_source=tour&utm_medium=${placement}`;
}
