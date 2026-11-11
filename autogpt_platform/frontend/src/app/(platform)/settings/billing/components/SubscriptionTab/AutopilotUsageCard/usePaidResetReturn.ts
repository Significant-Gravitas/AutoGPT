"use client";

import { useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV2GetCopilotUsageQueryKey,
  usePostV2ConfirmUsageReset,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { UsageResetConfirmResponse } from "@/app/api/__generated__/models/usageResetConfirmResponse";
import {
  CONFIRM_ATTEMPTS,
  CONFIRM_WAIT_MS,
  grantedMessage,
} from "@/app/(platform)/copilot/components/UsageLimits/PaidReset/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { formatExactReset } from "@/services/usageExperience/resetTime";

type Confirm = () => Promise<{ status: number; data: unknown }>;

function pause(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Ask whether the card payment went through, a few times: Stripe usually
 *  confirms within seconds of sending the customer back. */
async function confirmPayment(confirm: Confirm, waitMs: number) {
  for (let attempt = 0; attempt < CONFIRM_ATTEMPTS; attempt += 1) {
    if (attempt > 0) await pause(waitMs);
    const response = await confirm().catch(() => null);
    if (response?.status !== 200) continue;
    const answer = response.data as UsageResetConfirmResponse;
    if (answer.status === "granted") return answer;
  }
  return null;
}

async function reportReturn(confirm: Confirm, waitMs: number) {
  const granted = await confirmPayment(confirm, waitMs);
  if (granted) {
    toast({
      title: "Your limits are reset",
      description: grantedMessage(formatExactReset(granted.resets_at)),
    });
    return;
  }
  toast({
    title: "Confirming your payment",
    description:
      "Your limits reset as soon as the payment goes through. If it didn’t, nothing was charged.",
  });
}

/** Back from paying for a usage reset by card: confirm it, say what
 *  happened, and drop the query string. */
export function usePaidResetReturn(waitMs = CONFIRM_WAIT_MS) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const { mutateAsync: confirm } = usePostV2ConfirmUsageReset();
  const outcome = searchParams.get("usage_reset");
  const sessionId = searchParams.get("session_id");
  const handled = useRef<string | null>(null);

  useEffect(
    function handlePaidResetReturn() {
      if (!outcome) return;
      // The page re-renders after the redirect: handle each return once.
      if (handled.current === outcome) return;
      handled.current = outcome;
      if (outcome === "success" && sessionId) {
        void reportReturn(
          () => confirm({ data: { session_id: sessionId } }),
          waitMs,
        ).then(() =>
          queryClient.invalidateQueries({
            queryKey: getGetV2GetCopilotUsageQueryKey(),
          }),
        );
      } else if (outcome === "cancel") {
        toast({
          title: "Payment cancelled",
          description: "Your limits weren’t reset and nothing was charged.",
        });
      }
      router.replace("/settings/billing");
    },
    [outcome, sessionId, confirm, queryClient, router, waitMs],
  );
}
