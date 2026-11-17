import type { LinkPurchaseApproval } from "@/app/api/__generated__/models/linkPurchaseApproval";

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

// A line for the person beside the status. The tool's own message is
// written for Otto, tool names and all, so the card never shows it.
const STATUS_DETAILS: Record<string, string> = {
  created: "Approve it in Link, then continue here.",
  pending_approval: "Approve it in Link, then continue here.",
  declined: "Nothing was charged.",
  denied: "Nothing was charged.",
  expired: "Nothing was charged.",
  not_submitted: "Nothing was charged.",
  canceled: "Nothing was charged.",
  outcome_unknown: "Don't pay again until Link shows what happened.",
};

// Hosts Link sends a customer to for approval or a required action; the
// backend applies the same rule.
const LINK_ACTION_DOMAINS = ["link.com", "stripe.com"];

// Payment pages many sellers share: their address alone doesn't say who gets
// paid.
const SHARED_CHECKOUT_HOSTS = [
  "buy.stripe.com",
  "checkout.stripe.com",
  "paypal.com",
  "www.paypal.com",
  "square.link",
  "checkout.square.site",
  "buy.paddle.com",
];

// Only these tools return a checkout; another tool's output that claims the
// type must not get a card with Approve on it.
const CHECKOUT_TOOLS = [
  "browser_request_link_payment",
  "browser_raise_link_payment",
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
    address: purchaseAddress(output.merchant_url),
    sharedPage: isSharedCheckoutPage(output.merchant_url),
    total: formatTotal(output.amount, output.currency),
    approvalUrl: showsLink ? trustedLinkUrl(link) : "",
    linkLabel:
      key === "requires_action"
        ? "Complete verification in Link"
        : "Review in Link",
    status: STATUS_LABELS[key] ?? "Link checkout",
    detail: STATUS_DETAILS[key] ?? "",
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

export function purchaseDetails(record: LinkPurchaseApproval) {
  return {
    merchant: record.merchant_name,
    address: purchaseAddress(record.merchant_url),
    sharedPage: isSharedCheckoutPage(record.merchant_url),
    total: formatTotal(record.amount, record.currency),
    previousTotal:
      typeof record.previous_amount === "number"
        ? formatTotal(record.previous_amount, record.currency)
        : "",
    reason: record.reason ?? "",
    context: record.context,
    testMode: record.test_mode,
  };
}

export function purchaseAddress(merchantUrl: unknown) {
  if (typeof merchantUrl !== "string") return "";
  try {
    const url = new URL(merchantUrl);
    return url.hostname + (url.pathname === "/" ? "" : url.pathname);
  } catch {
    return "";
  }
}

export function isSharedCheckoutPage(merchantUrl: unknown) {
  if (typeof merchantUrl !== "string") return false;
  try {
    return SHARED_CHECKOUT_HOSTS.includes(new URL(merchantUrl).hostname);
  } catch {
    return false;
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
