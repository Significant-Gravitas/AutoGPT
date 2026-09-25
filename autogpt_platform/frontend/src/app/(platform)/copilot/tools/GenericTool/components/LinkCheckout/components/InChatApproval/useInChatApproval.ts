import {
  useGetV2GetALinkPurchaseApproval,
  usePostV2ApproveALinkPurchase,
  usePostV2DeclineALinkPurchase,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { CopilotChatActionsContext } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/useCopilotChatActions";
import { useContext, useState } from "react";
import {
  continueAfterDecision,
  formatTotal,
  purchaseHost,
} from "../../helpers";

interface Args {
  sessionId: string;
  checkoutId: string;
}

export function useInChatApproval({ sessionId, checkoutId }: Args) {
  const actions = useContext(CopilotChatActionsContext);
  const readOnly = !actions || actions.chatSurface === "share";
  const [error, setError] = useState("");
  // The purchase as the server recorded it, not as the transcript tells it:
  // this is what the customer approves.
  const approval = useGetV2GetALinkPurchaseApproval(sessionId, checkoutId, {
    query: {
      // A shared chat's viewer is not the customer and cannot read it.
      enabled: !readOnly && !!sessionId && !!checkoutId,
      select: (response) => (response.status === 200 ? response.data : null),
    },
  });
  const approve = usePostV2ApproveALinkPurchase();
  const decline = usePostV2DeclineALinkPurchase();
  const pending = approve.isPending || decline.isPending;
  const record = approval.data;

  async function decide(approved: boolean) {
    if (readOnly || pending) return;
    setError("");
    try {
      await (approved ? approve : decline).mutateAsync({
        sessionId,
        checkoutId,
      });
    } catch {
      setError(
        "Your decision was not recorded. The purchase is shown as it stands.",
      );
      await approval.refetch();
      return;
    }
    await approval.refetch();
    await actions?.onSend(continueAfterDecision({ checkoutId, approved }));
  }

  return {
    readOnly,
    state: record?.state,
    purchase: record && {
      merchant: record.merchant_name,
      host: purchaseHost(record.merchant_url),
      total: formatTotal(record.amount, record.currency),
      context: record.context,
      testMode: record.test_mode,
    },
    isLoading: approval.isLoading,
    approving: approve.isPending,
    declining: decline.isPending,
    error,
    handleApprove: () => decide(true),
    handleDecline: () => decide(false),
  };
}
