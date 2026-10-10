import { costRowSchema } from "./catalog-connected";
import { asNumber } from "./calculations";

type ActionValue =
  | string
  | number
  | boolean
  | null
  | ActionValue[]
  | { [key: string]: ActionValue };

export function getActionFields(state: unknown, formName?: string) {
  if (!state || typeof state !== "object") return {};
  const record = state as Record<string, unknown>;
  const fields = formName ? record[formName] : record;
  if (!fields || typeof fields !== "object") return {};
  const result: Record<string, ActionValue> = {};
  for (const [name, field] of Object.entries(fields).slice(0, 32)) {
    if (name.endsWith("__valid")) continue;
    const value =
      field && typeof field === "object" && "value" in field
        ? field.value
        : field;
    const safe = actionValue(value);
    if (safe !== undefined)
      Object.defineProperty(result, name, { value: safe, enumerable: true });
  }
  return result;
}

function actionValue(value: unknown, depth = 0): ActionValue | undefined {
  if (depth > 4) return undefined;
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "string") return value.slice(0, 2000);
  if (typeof value === "number")
    return Number.isFinite(value) ? value : undefined;
  if (Array.isArray(value)) {
    if (value.length > 12) return undefined;
    const items = value.map((item) => actionValue(item, depth + 1));
    return items.every((item) => item !== undefined) ? items : undefined;
  }
  if (!value || typeof value !== "object" || Object.keys(value).length > 12)
    return undefined;
  const record = value as Record<string, unknown>;
  const cost = costRowSchema.safeParse({
    ...record,
    quantity: asNumber(record.quantity),
    unitPrice: asNumber(record.unitPrice),
  });
  if (cost.success) return cost.data;
  if (
    typeof record.id === "string" &&
    typeof record.label === "string" &&
    record.included === false &&
    "quantity" in record &&
    "unitPrice" in record
  ) {
    return {
      id: record.id.slice(0, 200),
      label: record.label.slice(0, 200),
      included: false,
    };
  }
  return Object.fromEntries(
    Object.entries(record).flatMap(([key, item]) => {
      const safe = actionValue(item, depth + 1);
      return safe === undefined ? [] : [[key.slice(0, 200), safe]];
    }),
  );
}
