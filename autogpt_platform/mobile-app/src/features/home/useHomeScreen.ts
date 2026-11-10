import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { chatKeys, getSuggestedPrompts } from "@/lib/api/chat";

import { startChat } from "./startChat";

const FALLBACK_PROMPTS = [
  "Plan my week",
  "Draft a follow-up email",
  "Research a topic for me",
  "Sum up a long document",
];

export function useHomeScreen() {
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { data } = useQuery({
    queryKey: chatKeys.suggestedPrompts,
    queryFn: getSuggestedPrompts,
  });

  const suggested = (data ?? []).filter(Boolean).slice(0, 6);
  const prompts = suggested.length > 0 ? suggested : FALLBACK_PROMPTS;

  async function submit(text: string) {
    if (isStarting) return;
    setIsStarting(true);
    setError(null);
    try {
      await startChat(text);
    } catch {
      setError("Couldn't start a chat. Try again.");
    } finally {
      setIsStarting(false);
    }
  }

  return { prompts, isStarting, error, submit };
}
