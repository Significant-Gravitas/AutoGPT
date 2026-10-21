"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useTrialCheckoutReturn } from "@/services/trials/useTrialCheckoutReturn";

export function TrialCheckoutConfirmation() {
  const confirmation = useTrialCheckoutReturn();
  if (confirmation.error)
    return (
      <ErrorCard
        context="your trial"
        responseError={{ message: confirmation.error }}
        onRetry={confirmation.retry}
      />
    );
  if (!confirmation.ready)
    return (
      <Text variant="body" role="status">
        Confirming your trial and card setup…
      </Text>
    );
  return null;
}
