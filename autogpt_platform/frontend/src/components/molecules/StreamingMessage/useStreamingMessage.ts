import { useEffect, useState } from "react";

interface UseStreamingMessageArgs {
  chunks: string[];
  onComplete?: () => void;
}

interface UseStreamingMessageResult {
  displayText: string;
  isComplete: boolean;
}

export function useStreamingMessage({
  chunks,
  onComplete,
}: UseStreamingMessageArgs): UseStreamingMessageResult {
  const [isComplete, _setIsComplete] = useState(false);

  // Accumulate all chunks into display text
  const displayText = chunks.join("");

  // Detect completion (no new chunks for a while, or explicit done signal)
  useEffect(
    function detectCompletion() {
      // For now, we'll rely on the parent to signal completion
      // This hook mainly serves to manage the streaming state
      if (isComplete && onComplete) {
        onComplete();
      }
    },
    [isComplete, onComplete],
  );

  return {
    displayText,
    isComplete,
  };
}
