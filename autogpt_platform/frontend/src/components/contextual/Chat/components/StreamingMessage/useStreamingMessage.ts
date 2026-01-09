import { useEffect, useState } from "react";

interface UseStreamingMessageArgs {
  chunks: string[];
  onComplete?: () => void;
}

export function useStreamingMessage({
  chunks,
  onComplete,
}: UseStreamingMessageArgs) {
  const [isComplete, _setIsComplete] = useState(false);
  const displayText = chunks.join("");

  useEffect(() => {
    if (isComplete && onComplete) {
      onComplete();
    }
  }, [isComplete, onComplete]);

  return {
    displayText,
    isComplete,
  };
}
