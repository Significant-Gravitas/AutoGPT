import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import { parseFormula } from "./formula";

const webURL = z.url({ protocol: /^https?$/ }).max(2000);
const shortText = z.string().min(1).max(200);

export const Comparison = defineComponent({
  name: "Comparison",
  description:
    "Two selectable comparison cards inside Form. On mobile, swipe left chooses the first option and right the second; buttons and undo also work. Selection stays local until Form submission. value is an option id, or empty for no choice. Facts, images and source links must be supplied or verified. Optional amount is a comparable numeric value in the SAME unit for both options; it is published as name_amount, with name_label for the title. Do not invent prices or use 0 for unknown amounts.",
  props: z
    .object({
      name: shortText,
      label: shortText,
      value: z.string(),
      options: z
        .array(
          z.object({
            id: shortText,
            title: shortText,
            description: z.string().max(500),
            facts: z
              .array(z.object({ label: shortText, value: z.string().max(300) }))
              .max(6),
            amount: z.number().optional(),
            image: z.object({ url: webURL, alt: shortText }).optional(),
            source: z.object({ url: webURL, label: shortText }).optional(),
          }),
        )
        .length(2),
    })
    .refine(
      ({ options, value }) =>
        new Set(options.map((option) => option.id)).size === 2 &&
        (!value || options.some((option) => option.id === value)),
      {
        message:
          "Use two distinct option ids and an empty or matching selected value.",
      },
    ),
  component: null,
});

export const costRowSchema = z.object({
  id: shortText,
  label: shortText,
  quantity: z.number().min(0).max(1_000_000),
  unitPrice: z.number().min(0).max(1_000_000_000),
  included: z.boolean(),
});

export const CostTable = defineComponent({
  name: "CostTable",
  description:
    "Editable line items inside Form: quantities, unit prices, and include/exclude controls. Up to eight rows with unique ids. Recalculates included row totals locally; exposes the rows at name and a numeric name_total to CalculatedMetric. Row quantities are independent: they do not follow a separate NumberField. For charges driven by a shared headcount or duration, use that NumberField directly in CalculatedMetric instead of copying its initial value into a row. One ISO currency per table. Unknown prices must be clarified, never entered as 0. Use this for a budget or estimate, not arbitrary datasets. It does not purchase anything.",
  props: z
    .object({
      name: shortText,
      title: shortText,
      currency: z.string().regex(/^[A-Z]{3}$/),
      rows: z.array(costRowSchema).min(1).max(8),
    })
    .refine(
      ({ rows }) => new Set(rows.map((row) => row.id)).size === rows.length,
      { message: "CostTable row ids must be unique." },
    ),
  component: null,
});

export const CalculatedMetric = defineComponent({
  name: "CalculatedMetric",
  description:
    "A summary that updates locally from fields in the named Form. formula uses exact numeric field names, constants, + - * / and parentheses; count(interests) counts a MultiSelectField. Examples: stay_amount * nights + extras_total; (stay_amount + breakfast_rate) * nights + booking_fee. Reuse the same quantity field for every dependent charge, and keep flat fees outside that multiplication. Verify both choices and a changed quantity. Use NumberField names, Comparison's name_amount and CostTable's name_total. No JavaScript or metric-to-metric references. Incomplete/invalid inputs pause the result. format is number, currency (unit=ISO currency), or percent (ratio 0.5 => 50%). Never use static Metric for values that should recalculate.",
  props: z
    .object({
      form: shortText,
      label: shortText,
      formula: z
        .string()
        .max(500)
        .superRefine((formula, context) => {
          try {
            parseFormula(formula);
          } catch (error) {
            context.addIssue({
              code: "custom",
              message:
                error instanceof Error ? error.message : "Invalid formula.",
            });
          }
        }),
      format: z.enum(["number", "currency", "percent"]),
      unit: z.string().max(30),
      precision: z.number().int().min(0).max(4),
    })
    .refine(
      ({ format, unit }) => format !== "currency" || /^[A-Z]{3}$/.test(unit),
      { message: "Use an ISO currency code for currency formatting." },
    ),
  component: null,
});
