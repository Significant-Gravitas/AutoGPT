import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { PLAN_REQUIRED_MESSAGE } from "@/features/chat/transport";
import { chatKeys, getSuggestedPrompts } from "@/lib/api/chat";
import { ApiError } from "@/lib/api/client";

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
  // State lands a render late; the ref stops a second tap in the same frame.
  const isStartingRef = useRef(false);
  const { data } = useQuery({
    queryKey: chatKeys.suggestedPrompts,
    queryFn: getSuggestedPrompts,
  });

  const suggested = (data ?? []).filter(Boolean).slice(0, 6);
  const prompts = suggested.length > 0 ? suggested : FALLBACK_PROMPTS;

  async function submit(text: string) {
    if (isStartingRef.current) return false;
    isStartingRef.current = true;
    setIsStarting(true);
    setError(null);
    try {
      await startChat(text);
      return true;
    } catch (caught) {
      setError(
        caught instanceof ApiError && caught.status === 402
          ? PLAN_REQUIRED_MESSAGE
          : "Couldn't start a chat. Try again.",
      );
      return false;
    } finally {
      isStartingRef.current = false;
      setIsStarting(false);
    }
  }

  return { prompts, isStarting, error, submit };
}
