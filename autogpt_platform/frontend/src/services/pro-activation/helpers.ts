export function formatAmount(amount: number, currency: string) {
  const formatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: currency.toUpperCase(),
  });
  const digits = formatter.resolvedOptions().maximumFractionDigits ?? 2;
  return formatter.format(amount / 10 ** digits);
}

export function intervalLabel(interval: string, count: number) {
  return count === 1 ? interval : `${count} ${interval}s`;
}

export function safeReturnTo(path: string) {
  return path.startsWith("/") &&
    !path.startsWith("//") &&
    !path.includes("\\") &&
    !/[\u0000-\u001f]/.test(path)
    ? path
    : "/settings/billing";
}

export function invoiceURL(value?: string | null) {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" && url.hostname === "invoice.stripe.com"
      ? url.href
      : null;
  } catch {
    return null;
  }
}
