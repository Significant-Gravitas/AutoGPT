import {
  campaign,
  campaignPlan,
  failures,
  leads,
  outreach,
  performance,
} from "./samples";

export const scenarios = [
  {
    id: "performance",
    name: "Agent performance",
    label: "Understand your workspace",
    prompt: "How did my agents perform this week?",
    source: performance,
    suggestions: ["Investigate failed runs", "Show agent performance"],
  },
  {
    id: "leads",
    name: "Lead research",
    label: "Turn research into next steps",
    prompt: "Find promising customers for my automation product.",
    source: leads,
    suggestions: ["Create an outreach plan", "Show lead research"],
  },
  {
    id: "campaign",
    name: "Campaign planner",
    label: "Go from a brief to a plan",
    prompt: "Help me plan a launch for my new product.",
    source: campaign,
    suggestions: ["Show campaign brief", "Build my campaign plan"],
  },
] as const;

export type Scenario = (typeof scenarios)[number];

export function getSampleResponse(
  prompt: string,
  fields: Record<string, unknown> = {},
) {
  const query = prompt
    .trim()
    .toLowerCase()
    .replace(/[?.!]$/, "");
  if (query === "investigate failed runs") return failures;
  if (query === "create an outreach plan") return outreach;
  if (query === "build my campaign plan") return campaignPlan(fields);
  if (query === "show campaign brief") return campaign;
  if (query === "show agent performance") return performance;
  if (query === "show lead research") return leads;
  return (
    scenarios.find(
      (scenario) =>
        scenario.prompt.toLowerCase().replace(/[?.!]$/, "") === query,
    )?.source ?? null
  );
}
