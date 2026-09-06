import type { CreditTransactionItem } from "@/app/api/__generated__/models/creditTransactionItem";
import { formatCents } from "../../../helpers";

export type Transaction = CreditTransactionItem & {
  id: string;
  amount: number;
  transaction_type: NonNullable<CreditTransactionItem["transaction_type"]>;
  receipt_as_of?: string | Date | null;
};

const transactionNames: Record<string, string> = {
  TOP_UP: "Credits added",
  GRANT: "Credits granted",
  REFUND: "Top-up refunded",
  SUBSCRIPTION: "Subscription payment",
  CARD_CHECK: "Card verification",
};

export function activityName(transaction: Transaction): string {
  if (transaction.activity_type === "agent_run")
    return transaction.agent_name || "Agent unavailable";
  if (transaction.activity_type === "copilot_tools")
    return "Autopilot tool use";
  if (transaction.activity_type === "block_usage") return "Direct block usage";
  if (transaction.transaction_type === "USAGE")
    return transaction.description || "Credit usage";
  return (
    transactionNames[transaction.transaction_type] || "Credit balance change"
  );
}

export function activityDescription(transaction: Transaction): string {
  if (transaction.parent_agent_name)
    return `Part of ${transaction.parent_agent_name}`;
  if (transaction.activity_type === "agent_run") return "Agent run";
  if (transaction.activity_type === "copilot_tools")
    return transaction.conversation_title || "Paid block tools";
  if (transaction.activity_type === "block_usage")
    return "Outside an agent run";
  if (transaction.transaction_type === "TOP_UP") return "Credit purchase";
  if (transaction.transaction_type === "REFUND")
    return "Credit purchase refunded";
  if (transaction.transaction_type === "SUBSCRIPTION")
    return "Paid from credit balance";
  return "Credit balance";
}

export function formatAmount(amount: number): string {
  const sign = amount < 0 ? "−" : amount > 0 ? "+" : "";
  return `${sign}${formatCents(Math.abs(amount))}`;
}

export function formatActivityDate(value: string | Date | null | undefined) {
  if (!value)
    return { date: "Unavailable", time: "", full: "Date unavailable" };
  const date = new Date(value);
  if (Number.isNaN(date.getTime()))
    return { date: "Unavailable", time: "", full: "Date unavailable" };
  return {
    date: date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    }),
    time: date.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
    full: date.toLocaleString(undefined, {
      dateStyle: "long",
      timeStyle: "long",
    }),
  };
}

export function libraryHref(libraryID: string | null | undefined) {
  return libraryID ? `/library/agents/${encodeURIComponent(libraryID)}` : null;
}

export function taskHref(transaction: Transaction) {
  const agentHref = libraryHref(transaction.library_agent_id);
  if (
    !agentHref ||
    !transaction.execution_available ||
    !transaction.usage_execution_id
  )
    return null;
  return `${agentHref}?activeTab=runs&activeItem=${encodeURIComponent(transaction.usage_execution_id)}`;
}

export function isRunActive(transaction: Transaction) {
  return ["QUEUED", "RUNNING", "REVIEW", "INCOMPLETE"].includes(
    transaction.execution_status || "",
  );
}

export function statusLabel(status: string) {
  const labels: Record<string, string> = {
    QUEUED: "Queued",
    RUNNING: "Running",
    REVIEW: "Needs review",
    INCOMPLETE: "Incomplete",
    COMPLETED: "Completed",
    FAILED: "Failed",
    TERMINATED: "Stopped",
  };
  return labels[status] || "Unavailable";
}

export function receiptNote(transaction: Transaction) {
  if (transaction.activity_type === "agent_run") {
    if (isRunActive(transaction))
      return "This run is still active. Charges and adjustments may continue.";
    if (transaction.amount === 0)
      return "No net change to your credit balance.";
    return "Recorded charges and adjustments for this run.";
  }
  if (transaction.activity_type === "copilot_tools")
    return "Paid block tools across this conversation. Subscription usage is tracked separately.";
  if (transaction.activity_type === "block_usage")
    return "A paid block call without an associated agent run.";
  if (transaction.transaction_type === "REFUND")
    return "Credits removed from your balance after the payment was refunded.";
  if (transaction.transaction_type === "SUBSCRIPTION")
    return "This payment was taken from your credit balance.";
  if (transaction.amount === 0) return "No change to your credit balance.";
  return "";
}

export function chargeLabel(type: string) {
  const labels: Record<string, string> = {
    usage: "Block usage",
    execution_fee: "Execution fee",
    adjustment: "Usage adjustment",
    transaction: "Balance change",
  };
  return labels[type] || "Balance change";
}
