const STATUS_LABELS: Record<string, string> = {
  awaiting_approval: "Waiting for your approval",
  created: "Waiting for Link",
  pending_approval: "Waiting for your approval in Link",
  approved: "Approved",
  declined: "Purchase declined",
  denied: "Purchase declined",
  expired: "Approval expired",
  requires_action: "Link needs your attention",
  submitted: "Submitted once · confirmation pending",
  outcome_unknown: "Check payment status before taking any further action",
  not_submitted: "Checkout stopped",
  succeeded: "Payment confirmed by Link",
  failed: "Link reports the payment failed",
  canceled: "Purchase canceled",
  browser_reset: "Ready to browse again",
};

// Hosts Link sends a customer to for approval or a required action; the
// backend applies the same rule.
const LINK_ACTION_DOMAINS = ["link.com", "stripe.com"];

// Only these tools return a checkout; another tool's output that claims the
// type must not get a card with Approve on it.
const CHECKOUT_TOOLS = [
  "browser_request_link_payment",
  "browser_complete_link_payment",
  "browser_link_payment_status",
  "browser_reset_after_payment",
];

export function isLinkCheckoutOutput(
  toolName: string | undefined,
  output: Record<string, unknown> | null,
) {
  return (
    !!toolName &&
    CHECKOUT_TOOLS.includes(toolName) &&
    output?.type === "browser_checkout"
  );
}

export function checkoutPresentation(output: Record<string, unknown>) {
  const merchant =
    typeof output.merchant_name === "string"
      ? output.merchant_name
      : "Link checkout";
  const key = typeof output.status === "string" ? output.status : "";
  const link =
    key === "requires_action" ? output.action_url : output.approval_url;
  const showsLink = ["created", "pending_approval", "requires_action"].includes(
    key,
  );
  return {
    merchant,
    total: formatTotal(output.amount, output.currency),
    approvalUrl: showsLink ? trustedLinkUrl(link) : "",
    linkLabel:
      key === "requires_action"
        ? "Complete verification in Link"
        : "Review in Link",
    status: STATUS_LABELS[key] ?? "Link checkout",
    message: typeof output.message === "string" ? output.message : "",
    actionMessage:
      key === "requires_action" && typeof output.action_message === "string"
        ? output.action_message
        : "",
    testMode: output.test_mode === true,
    approvesInChat:
      output.approval_mode === "in_app" && key === "awaiting_approval",
  };
}

export function continueAfterDecision({
  checkoutId,
  approved,
}: {
  checkoutId: string;
  approved: boolean;
}) {
  if (!approved)
    return `I declined purchase ${checkoutId}. Don't buy it; ask me what to do instead.`;
  return `I approved purchase ${checkoutId} in the chat. Complete it now with run_capability id "tool:browser_complete_link_payment" and input {"checkout_id":"${checkoutId}"}.`;
}

export function purchaseHost(merchantUrl: string) {
  try {
    return new URL(merchantUrl).hostname;
  } catch {
    return "";
  }
}

export function formatTotal(amount: unknown, currency: unknown) {
  const code = typeof currency === "string" ? currency : "usd";
  if (typeof amount !== "number" || !/^[a-z]{3}$/.test(code)) return "";
  try {
    const formatter = new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: code,
    });
    const digits = formatter.resolvedOptions().maximumFractionDigits ?? 2;
    return formatter.format(amount / 10 ** digits);
  } catch {
    return `${amount} ${code.toUpperCase()} minor units`;
  }
}

function trustedLinkUrl(value: unknown) {
  if (typeof value !== "string") return "";
  try {
    const parsed = new URL(value);
    const host = parsed.hostname.toLowerCase();
    const trusted =
      parsed.protocol === "https:" &&
      !parsed.username &&
      !parsed.password &&
      !parsed.port &&
      LINK_ACTION_DOMAINS.some(
        (domain) => host === domain || host.endsWith(`.${domain}`),
      );
    return trusted ? parsed.href : "";
  } catch {
    return "";
  }
}
