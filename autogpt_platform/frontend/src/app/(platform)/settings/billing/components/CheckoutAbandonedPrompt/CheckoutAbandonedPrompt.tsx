"use client";

import { CheckoutAbandonedDialog } from "@/components/organisms/FeedbackPrompts/CheckoutAbandonedDialog";
import { useFeedbackPrompt } from "@/components/organisms/FeedbackPrompts/useFeedbackPrompt";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useState } from "react";

export type AbandonedCheckout = "trial" | "upgrade";

interface Props {
  kind: AbandonedCheckout;
}

export function CheckoutAbandonedPrompt({ kind }: Props) {
  const prompt = useFeedbackPrompt(
    kind === "trial" ? "trial_checkout" : "upgrade_checkout",
  );
  const [isClaimed, setIsClaimed] = useState(false);

  useMountEffect(() => {
    void prompt.claim().then((id) => {
      if (id) setIsClaimed(true);
    });
  });

  if (!isClaimed) return null;

  return (
    <CheckoutAbandonedDialog
      kind={kind}
      onReason={(reason) => prompt.save({ reason })}
      onSubmit={({ reason, note }) =>
        prompt.save({ reason, note: note || undefined })
      }
    />
  );
}
