"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useDelegationAnswerStore } from "./delegationAnswerStore";

type Delivery = "sent" | "drafted";

/** How an answer reaches the teammate's thread. There is no endpoint yet
 *  that posts into another session without streaming it, so the answer is
 *  drafted into that thread's composer for the user to send. Swap the body
 *  for the answer endpoint and return "sent" once it exists. */
async function deliverAnswer(
  subSessionId: string,
  text: string,
  navigate: (href: string) => void,
): Promise<Delivery> {
  const params = new URLSearchParams({
    sessionId: subSessionId,
    prefill: text,
  });
  navigate(`/copilot?${params.toString()}`);
  return "drafted";
}

/** Answers a teammate's question from Otto's chat or its Work panel. */
export function useDelegationAnswer() {
  const router = useRouter();
  const recordAnswer = useDelegationAnswerStore((s) => s.recordAnswer);
  const [isSending, setIsSending] = useState(false);

  async function sendAnswer(
    subSessionId: string,
    text: string,
    question: string | null,
  ) {
    const answer = text.trim();
    if (!answer || isSending) return;
    setIsSending(true);
    try {
      const delivery = await deliverAnswer(subSessionId, answer, (href) =>
        router.push(href),
      );
      if (delivery === "sent")
        recordAnswer(subSessionId, {
          question,
          text: answer,
          sentAt: Date.now(),
        });
    } catch {
      toast({
        title: "Couldn't send your answer",
        description: "Open the thread and answer there.",
        variant: "destructive",
      });
    } finally {
      setIsSending(false);
    }
  }

  return { sendAnswer, isSending };
}
