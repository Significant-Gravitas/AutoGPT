import type { User } from "@supabase/supabase-js";

export type PageState =
  | { type: "welcome" }
  | { type: "newChat" }
  | { type: "creating"; prompt: string }
  | { type: "chat"; sessionId: string; initialPrompt?: string };

export function getInitialPromptFromState(
  pageState: PageState,
  storedInitialPrompt: string | undefined,
) {
  if (storedInitialPrompt) return storedInitialPrompt;
  if (pageState.type === "creating") return pageState.prompt;
  if (pageState.type === "chat") return pageState.initialPrompt;
}

export function shouldResetToWelcome(pageState: PageState) {
  return (
    pageState.type !== "newChat" &&
    pageState.type !== "creating" &&
    pageState.type !== "welcome"
  );
}

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
