"use client";

import { CheckoutAbandonedDialog } from "@/components/organisms/FeedbackPrompts/CheckoutAbandonedDialog";
import { useFeedbackPrompt } from "@/components/organisms/FeedbackPrompts/useFeedbackPrompt";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useState } from "react";
import type { AbandonedPlan } from "./helpers";
import { useCheckoutRetry } from "./useCheckoutRetry";

interface Props {
  plan: AbandonedPlan | null;
}

export function CheckoutAbandonedPrompt({ plan }: Props) {
  const { user } = useAuth();
  const [ownerID, setOwnerID] = useState(user?.id);
  if (!user) return null;
  if (ownerID === undefined) setOwnerID(user.id);
  if (ownerID !== undefined && ownerID !== user.id) return null;
  return <CheckoutClaim plan={plan} />;
}

function CheckoutClaim({ plan }: Props) {
  const prompt = useFeedbackPrompt("upgrade_checkout");
  const checkout = useCheckoutRetry(plan);
  const [isClaimed, setIsClaimed] = useState(false);

  useMountEffect(() => {
    void prompt.claim().then(({ id }) => {
      if (id) setIsClaimed(true);
    });
  });

  if (!isClaimed) return null;

  return (
    <CheckoutAbandonedDialog
      isRetrying={checkout.isRetrying}
      onDismiss={() => prompt.save({ dismissed: true })}
      onReason={(reason) => prompt.save({ reason })}
      onSubmit={({ reason, note }) =>
        prompt.save({ reason, note: note || undefined })
      }
      onTryAnotherCard={checkout.retry}
    />
  );
}
