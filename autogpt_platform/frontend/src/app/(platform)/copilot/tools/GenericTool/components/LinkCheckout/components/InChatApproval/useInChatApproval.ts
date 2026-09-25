import {
  useGetV2GetALinkPurchaseApproval,
  usePostV2ApproveALinkPurchase,
  usePostV2DeclineALinkPurchase,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { CopilotChatActionsContext } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/useCopilotChatActions";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useContext, useState } from "react";
import { continueAfterDecision, purchaseDetails } from "../../helpers";

interface Args {
  sessionId: string;
  checkoutId: string;
}

export function useInChatApproval({ sessionId, checkoutId }: Args) {
  const actions = useContext(CopilotChatActionsContext);
  const readOnly = !actions || actions.chatSurface === "share";
  const [error, setError] = useState("");
  // A recorded decision whose message to the agent did not send.
  const [unsent, setUnsent] = useState<boolean | null>(null);
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
  const gone =
    approval.error instanceof ApiError && approval.error.status === 404;

  async function tellAgent(approved: boolean) {
    try {
      await actions?.onSend(continueAfterDecision({ checkoutId, approved }));
      setUnsent(null);
    } catch {
      setUnsent(approved);
    }
  }

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
    await tellAgent(approved);
  }

  return {
    readOnly,
    state: record?.state,
    purchase: record ? purchaseDetails(record) : null,
    isLoading: approval.isLoading,
    loadFailed: approval.isError && !gone,
    retryLoad: () => approval.refetch(),
    agentNotTold: unsent !== null,
    retryTellAgent: () => unsent !== null && tellAgent(unsent),
    approving: approve.isPending,
    declining: decline.isPending,
    error,
    handleApprove: () => decide(true),
    handleDecline: () => decide(false),
  };
}
