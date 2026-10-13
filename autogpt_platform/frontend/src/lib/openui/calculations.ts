export function asNumber(value: unknown) {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (
    typeof value !== "string" ||
    !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim())
  )
    return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function formatCalculated(
  value: number,
  format: string,
  unit: string,
  precision: number,
) {
  const options: Intl.NumberFormatOptions = {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  };
  if (format === "currency") {
    const currency = new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: unit,
    }).resolvedOptions();
    const digits = Math.max(precision, currency.maximumFractionDigits ?? 2);
    Object.assign(options, {
      style: "currency",
      currency: unit,
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }
  if (format === "percent") options.style = "percent";
  return (
    new Intl.NumberFormat("en-US", options).format(value) +
    (format === "number" && unit ? ` ${unit}` : "")
  );
}
