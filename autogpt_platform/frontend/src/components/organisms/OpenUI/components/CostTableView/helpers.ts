import { z } from "zod/v4";
import { costRowSchema } from "@/lib/openui/catalog-connected";
import { asNumber } from "@/lib/openui/calculations";

const draftRowSchema = costRowSchema.extend({
  quantity: z.union([z.number(), z.string().max(100)]),
  unitPrice: z.union([z.number(), z.string().max(100)]),
});
export type CostRowDraft = z.infer<typeof draftRowSchema>;

export function readCostRows(value: unknown, defaults: CostRowDraft[]) {
  const parsed = z.array(draftRowSchema).max(8).safeParse(value);
  if (!parsed.success || parsed.data.length !== defaults.length)
    return defaults;
  return defaults.map((row) => {
    const saved = parsed.data.find((item) => item.id === row.id);
    return saved ? { ...saved, label: row.label } : row;
  });
}

export function costInputError(value: unknown, kind: "quantity" | "unitPrice") {
  const number = asNumber(value);
  if (number === null) return "Enter a number, such as 10 or 10.5.";
  if (number < 0) return "Enter 0 or more.";
  if (number > (kind === "quantity" ? 1_000_000 : 1_000_000_000))
    return "This value is too large.";
  if (
    kind === "unitPrice" &&
    Math.abs(number * 100 - Math.round(number * 100)) > 0.00001
  )
    return "Use at most two decimal places.";
  return "";
}

export function rowTotal(row: CostRowDraft) {
  if (!row.included) return 0;
  if (
    costInputError(row.quantity, "quantity") ||
    costInputError(row.unitPrice, "unitPrice")
  )
    return null;
  return Math.round(Number(row.quantity) * Number(row.unitPrice) * 100) / 100;
}

export function costTotal(rows: CostRowDraft[]) {
  const totals = rows.map(rowTotal);
  if (totals.some((value) => value === null)) return null;
  return (
    Math.round(
      totals.reduce<number>((sum, value) => sum + (value ?? 0), 0) * 100,
    ) / 100
  );
}
