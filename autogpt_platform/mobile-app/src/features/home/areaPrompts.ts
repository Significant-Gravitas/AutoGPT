import type { Expert } from "@/lib/api/experts";
import { expertFamily, type Family } from "@/lib/expertPalette";

// Starters for a chat with an expert, by the expert's area. Each one is the
// message sent to that expert, so it reads as a request to them.
const AREA_PROMPTS: Record<Family, string[]> = {
  marketing: [
    "Plan this month's content calendar",
    "Draft three posts for our launch",
    "Review our landing page copy",
    "Find what competitors are posting",
  ],
  sales: [
    "Find 10 leads that fit our best customers",
    "Draft a cold outreach email",
    "Prep me for my next sales call",
    "Follow up on deals that went quiet",
  ],
  finance: [
    "Sum up last month's spending",
    "Build a simple cash flow forecast",
    "Chase unpaid invoices",
    "Flag unusual expenses",
  ],
  support: [
    "Draft replies to open tickets",
    "Find this week's top customer complaints",
    "Write a help article for a common question",
    "Sum up feedback from recent reviews",
  ],
  operations: [
    "Map out a process I repeat every week",
    "Plan next week's priorities",
    "Write a checklist for onboarding a new hire",
    "Find tasks I could hand off",
  ],
  research: [
    "Research a market and sum up the findings",
    "Compare our top three competitors",
    "Find recent studies on a topic",
    "Track news about our industry",
  ],
  content: [
    "Draft a blog post outline",
    "Turn a long document into a short summary",
    "Rewrite this copy to be clearer",
    "Plan the next newsletter",
  ],
  development: [
    "Review my latest pull request",
    "Triage flaky tests",
    "Write a pre-release checklist",
    "Draft an incident postmortem",
  ],
  general: [
    "Plan my week",
    "Draft a follow-up email",
    "Research a topic for me",
    "Sum up a long document",
  ],
  otto: [
    "Plan my week",
    "Draft a follow-up email",
    "Research a topic for me",
    "Sum up a long document",
  ],
};

export function areaPrompts(
  expert: Pick<Expert, "avatar_url" | "categories" | "role">,
) {
  const family = expertFamily({
    avatarUrl: expert.avatar_url,
    categories: expert.categories,
    role: expert.role,
  });
  return AREA_PROMPTS[family];
}
