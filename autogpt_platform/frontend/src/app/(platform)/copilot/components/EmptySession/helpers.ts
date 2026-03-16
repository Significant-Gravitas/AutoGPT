import { User } from "@supabase/supabase-js";

export function getInputPlaceholder(width?: number) {
  if (!width) return "What's your role and what eats up most of your day?";

  if (width < 500) {
    return "I'm a chef and I hate...";
  }
  if (width <= 1080) {
    return "What's your role and what eats up most of your day?";
  }
  return "What's your role and what eats up most of your day? e.g. 'I'm a recruiter and I hate...'";
}

export interface SuggestionTheme {
  name: string;
  prompts: string[];
}

export const DEFAULT_THEMES: SuggestionTheme[] = [
  {
    name: "Learn",
    prompts: [
      "What can AutoGPT do for me?",
      "Show me how agents work",
      "What integrations are available?",
      "How do I schedule an agent?",
      "What are the most popular agents?",
    ],
  },
  {
    name: "Create",
    prompts: [
      "Draft a weekly status report",
      "Generate social media posts for my business",
      "Create a competitive analysis summary",
      "Write onboarding emails for new hires",
      "Build a content calendar for next month",
    ],
  },
  {
    name: "Automate",
    prompts: [
      "Monitor my competitors' websites for changes",
      "Send me a daily news digest on my industry",
      "Auto-reply to common customer questions",
      "Track price changes on products I sell",
      "Summarize my emails every morning",
    ],
  },
  {
    name: "Organize",
    prompts: [
      "Sort my bookmarks into categories",
      "Create a project timeline from my notes",
      "Prioritize my task list by urgency",
      "Build a decision matrix for vendor selection",
      "Organize my meeting notes into action items",
    ],
  },
];

export function getSuggestionThemes(
  apiThemes?: SuggestionTheme[],
): SuggestionTheme[] {
  if (apiThemes && apiThemes.length > 0) {
    return apiThemes;
  }
  return DEFAULT_THEMES;
}

export function getGreetingName(user?: User | null) {
  if (!user) return "there";
  const metadata = user.user_metadata as Record<string, unknown> | undefined;
  const fullName = metadata?.full_name;
  const name = metadata?.name;
  if (typeof fullName === "string" && fullName.trim()) {
    return fullName.split(" ")[0];
  }
  if (typeof name === "string" && name.trim()) {
    return name.split(" ")[0];
  }
  if (user.email) {
    return user.email.split("@")[0];
  }
  return "there";
}
