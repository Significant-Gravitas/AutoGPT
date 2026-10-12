"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { useAnswerSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useDelegationAnswerStore } from "./delegationAnswerStore";

type Delivery = "sent" | "drafted";

function draftHref(subSessionId: string, text: string) {
  const params = new URLSearchParams({
    sessionId: subSessionId,
    prefill: text,
  });
  return `/copilot?${params.toString()}`;
}

/** Answers a teammate's question from Otto's chat or its Work panel. The
 *  answer is posted straight into the teammate's thread; when that fails it
 *  is drafted into the thread's composer instead, for the user to send. */
export function useDelegationAnswer() {
  const router = useRouter();
  const recordAnswer = useDelegationAnswerStore((s) => s.recordAnswer);
  const { mutateAsync: postAnswer } = useAnswerSession();
  const [isSending, setIsSending] = useState(false);

  async function deliverAnswer(
    subSessionId: string,
    text: string,
  ): Promise<Delivery> {
    try {
      const response = await postAnswer({
        sessionId: subSessionId,
        data: { message: text },
      });
      if (response.status === 200) return "sent";
    } catch {
      // Fall through to the draft: the thread still takes the answer.
    }
    router.push(draftHref(subSessionId, text));
    return "drafted";
  }

  async function sendAnswer(
    subSessionId: string,
    text: string,
    question: string | null,
  ) {
    const answer = text.trim();
    if (!answer || isSending) return;
    setIsSending(true);
    try {
      const delivery = await deliverAnswer(subSessionId, answer);
      if (delivery === "sent")
        recordAnswer(subSessionId, {
          question,
          text: answer,
          sentAt: Date.now(),
        });
      else
        toast({
          title: "Couldn't send your answer",
          description: "It's drafted in the thread for you to send.",
          variant: "destructive",
        });
    } finally {
      setIsSending(false);
    }
  }

  return { sendAnswer, isSending };
}
