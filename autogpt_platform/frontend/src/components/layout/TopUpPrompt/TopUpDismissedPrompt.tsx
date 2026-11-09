"use client";

import { TopUpDismissedCard } from "@/components/organisms/FeedbackPrompts/TopUpDismissedCard";
import { useFeedbackPrompt } from "@/components/organisms/FeedbackPrompts/useFeedbackPrompt";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useState } from "react";

interface Props {
  onClosed: () => void;
}

export function TopUpDismissedPrompt({ onClosed }: Props) {
  const prompt = useFeedbackPrompt("top_up");
  const [isClaimed, setIsClaimed] = useState(false);

  useMountEffect(() => {
    void prompt.claim().then(({ id }) => {
      if (id) setIsClaimed(true);
      else onClosed();
    });
  });

  if (!isClaimed) return null;

  return (
    <div className="fixed right-6 bottom-6 z-40">
      <TopUpDismissedCard
        onReason={(reason) => prompt.save({ reason })}
        onSubmit={({ reason, note }) =>
          prompt.save({ reason, note: note || undefined })
        }
        onDismiss={() => prompt.save({ dismissed: true })}
        onClose={onClosed}
      />
    </div>
  );
}
