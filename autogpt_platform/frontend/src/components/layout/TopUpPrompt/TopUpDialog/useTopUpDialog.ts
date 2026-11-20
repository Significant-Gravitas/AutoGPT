import { useFeedbackPrompt } from "@/components/organisms/FeedbackPrompts/useFeedbackPrompt";
import { useRef, useState } from "react";

interface Args {
  onClose: () => void;
}

// "Not now" asks what would make topping up worth it, when the feedback limits
// allow a question (at most every 30 days); otherwise it just closes.
export function useTopUpDialog({ onClose }: Args) {
  const prompt = useFeedbackPrompt("top_up");
  const [isAsking, setIsAsking] = useState(false);
  const [isClaiming, setIsClaiming] = useState(false);
  const [isAnswered, setIsAnswered] = useState(false);
  // A claim that lands after the dialog closed must not reopen it on a question.
  const attemptRef = useRef(0);

  function close() {
    attemptRef.current += 1;
    if (isAsking && !isAnswered) prompt.save({ dismissed: true });
    setIsAsking(false);
    setIsClaiming(false);
    setIsAnswered(false);
    onClose();
  }

  async function notNow() {
    const attempt = ++attemptRef.current;
    setIsClaiming(true);
    const { id } = await prompt.claim(true);
    if (attempt !== attemptRef.current) return;
    setIsClaiming(false);
    if (id) setIsAsking(true);
    else close();
  }

  function answer({ reason, note }: { reason: string; note: string }) {
    setIsAnswered(true);
    prompt.save({ reason, note: note || undefined });
  }

  return {
    isAsking,
    isClaiming,
    close,
    notNow,
    pickReason: (reason: string) => prompt.save({ reason }),
    answer,
  };
}
