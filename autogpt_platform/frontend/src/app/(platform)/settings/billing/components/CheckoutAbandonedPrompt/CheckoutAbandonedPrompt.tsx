"use client";

import { CheckoutAbandonedDialog } from "@/components/organisms/FeedbackPrompts/CheckoutAbandonedDialog";
import { useFeedbackPrompt } from "@/components/organisms/FeedbackPrompts/useFeedbackPrompt";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useState } from "react";

export type AbandonedCheckout = "trial" | "upgrade";

interface Props {
  kind: AbandonedCheckout;
}

export function CheckoutAbandonedPrompt({ kind }: Props) {
  const { user } = useAuth();
  const [ownerID, setOwnerID] = useState(user?.id);
  if (!user) return null;
  if (ownerID === undefined) setOwnerID(user.id);
  if (ownerID !== undefined && ownerID !== user.id) return null;
  return <CheckoutClaim kind={kind} />;
}

function CheckoutClaim({ kind }: Props) {
  const prompt = useFeedbackPrompt(
    kind === "trial" ? "trial_checkout" : "upgrade_checkout",
  );
  const [isClaimed, setIsClaimed] = useState(false);

  useMountEffect(() => {
    void prompt.claim().then(({ id }) => {
      if (id) setIsClaimed(true);
    });
  });

  if (!isClaimed) return null;

  return (
    <CheckoutAbandonedDialog
      kind={kind}
      onDismiss={() => prompt.save({ dismissed: true })}
      onReason={(reason) => prompt.save({ reason })}
      onSubmit={({ reason, note }) =>
        prompt.save({ reason, note: note || undefined })
      }
    />
  );
}
