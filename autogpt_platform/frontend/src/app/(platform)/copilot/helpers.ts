import type { User } from "@supabase/supabase-js";

export function getGreetingName(user?: User | null): string {
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

export function buildCopilotChatUrl(prompt: string): string {
  const trimmed = prompt.trim();
  if (!trimmed) return "/copilot/chat";
  const encoded = encodeURIComponent(trimmed);
  return `/copilot/chat?prompt=${encoded}`;
}

export function getQuickActions(): string[] {
  return [
    "I don't know where to start, just ask me stuff",
    "I do the same thing every week and it's killing me",
    "Help me find where I'm wasting my time",
  ];
}

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
