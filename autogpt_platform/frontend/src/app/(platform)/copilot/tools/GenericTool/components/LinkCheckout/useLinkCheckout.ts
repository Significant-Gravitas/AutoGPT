import { CopilotChatActionsContext } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/useCopilotChatActions";
import { useContext, useState } from "react";

const CHECKOUT_ID = /^[a-f0-9]{32}$/;
// Before an attempt, asking the agent to continue checks Link's approval.
const APPROVAL_STATUSES = ["created", "pending_approval", "approved"];
// Once attempted, or while Link needs an action, only a status check is safe.
const STATUS_ONLY_STATUSES = [
  "requires_action",
  "submitted",
  "outcome_unknown",
  "not_submitted",
];

export function useLinkCheckout(output: Record<string, unknown>) {
  const actions = useContext(CopilotChatActionsContext);
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");
  const checkoutId =
    typeof output.checkout_id === "string" &&
    CHECKOUT_ID.test(output.checkout_id)
      ? output.checkout_id
      : "";
  const sessionId =
    typeof output.session_id === "string" ? output.session_id : "";
  const status = String(output.status);
  const readOnly = actions?.chatSurface === "share";
  const checkOnly =
    output.attempted === true || STATUS_ONLY_STATUSES.includes(status);
  const canContinue =
    !!actions &&
    !readOnly &&
    !!checkoutId &&
    !sent &&
    [...APPROVAL_STATUSES, ...STATUS_ONLY_STATUSES].includes(status);

  async function handleContinue() {
    if (!canContinue || sending) return;
    setSending(true);
    setError("");
    try {
      await actions.onSend(
        checkOnly
          ? `Use run_capability with id "tool:browser_link_payment_status" and input {"checkout_id":"${checkoutId}"}. This is a read-only status check; do not retrieve a card, submit payment, or create another request.`
          : `Check Link approval using run_capability with id "tool:browser_complete_link_payment" and input {"checkout_id":"${checkoutId}"}. Continue only if Link confirms approval; do not create another payment request.`,
      );
      setSent(true);
    } catch {
      setError("Could not send the approval check. Please try again.");
    } finally {
      setSending(false);
    }
  }

  return {
    readOnly,
    checkoutId,
    sessionId,
    canContinue,
    sending,
    error,
    buttonLabel: checkOnly
      ? "Check payment status"
      : "Check approval & continue",
    handleContinue,
  };
}
