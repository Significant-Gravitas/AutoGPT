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
    "Show me what I can automate",
    "Design a custom workflow",
    "Help me with content creation",
  ];
}
