import { CreditTransactionType } from "@/lib/autogpt-server-api";

export function formatAmount(amount: number, type: CreditTransactionType) {
  const isPositive = type === CreditTransactionType.GRANT;
  const isNeutral = type === CreditTransactionType.TOP_UP;
  const color = isPositive
    ? "text-green-600"
    : isNeutral
      ? "text-blue-600"
      : "text-red-600";
  return <span className={color}>${Math.abs(amount / 100)}</span>;
}

export function formatType(type: CreditTransactionType) {
  const isGrant = type === CreditTransactionType.GRANT;
  const isPurchased = type === CreditTransactionType.TOP_UP;
  const isSpent = type === CreditTransactionType.USAGE;

  const displayText = type;
  let bgColor = "";

  if (isGrant) {
    bgColor = "bg-green-100 text-green-800";
  } else if (isPurchased) {
    bgColor = "bg-blue-100 text-blue-800";
  } else if (isSpent) {
    bgColor = "bg-red-100 text-red-800";
  }

  return (
    <span className={`rounded-full px-2 py-1 text-xs font-medium ${bgColor}`}>
      {displayText.valueOf()}
    </span>
  );
}

export function formatDate(date: Date) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "numeric",
    hour12: true,
  }).format(new Date(date));
}
