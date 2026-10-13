import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import {
  cellValue,
  columnsSchema,
  dataColumn,
  identifier,
  rowsSchema,
  shortText,
} from "./data-values";

export const ValidatedField = defineComponent({
  name: "ValidatedField",
  description:
    "Typed input inside Form: text, number, date, boolean, email, URL or select. Local warnings block invalid submissions. column has key, label, type, optional required/options/unit. value may be null when unknown. Never collect passwords, API keys or one-time codes; use native connections.",
  props: z.object({
    name: identifier,
    column: dataColumn.refine(
      (column) => !column.calculation,
      "Use DataGrid for calculated columns.",
    ),
    value: cellValue.optional(),
  }),
  component: null,
});

export const SearchSelect = defineComponent({
  name: "SearchSelect",
  description:
    "Searchable choices inside Form, up to 100 supplied/authorized options with stable id, label, detail. value is an array; multiple=false permits zero or one selection. Empty means undecided and required controls whether it blocks submission. Keep current selections when filtering. Never invent account records.",
  props: z
    .object({
      name: identifier,
      label: shortText,
      options: z
        .array(
          z.object({
            id: identifier,
            label: shortText,
            detail: z.string().max(500),
          }),
        )
        .min(1)
        .max(100),
      value: z.array(identifier).max(100),
      multiple: z.boolean(),
      required: z.boolean(),
    })
    .refine(
      ({ options, value, multiple }) =>
        new Set(options.map((option) => option.id)).size === options.length &&
        new Set(value).size === value.length &&
        value.every((id) => options.some((option) => option.id === id)) &&
        (multiple || value.length <= 1),
      { message: "Use unique options and valid selections." },
    ),
  component: null,
});

export const FieldGroup = defineComponent({
  name: "FieldGroup",
  description:
    "Conditional/repeatable structured fields inside Form. columns describe typed inputs; rows contain stable id and cells. repeatable allows add/remove up to 20 groups; otherwise provide one row. visibleWhen optionally references another field in this same Form and an equals boolean/string. Hidden groups submit an empty array and cannot block validation. Their drafts are retained locally for toggling back on, not submitted. Use for attendee details or optional overnight preferences.",
  props: z
    .object({
      name: identifier,
      label: shortText,
      columns: columnsSchema.refine(
        (columns) => columns.every((column) => !column.calculation),
        "Use DataGrid for calculated columns.",
      ),
      rows: rowsSchema,
      repeatable: z.boolean(),
      required: z.boolean(),
      visibleWhen: z
        .object({
          field: identifier,
          equals: z.union([z.string(), z.boolean()]),
        })
        .optional(),
    })
    .refine(
      ({ rows, repeatable }) =>
        rows.length <= 20 && (repeatable || rows.length === 1),
      {
        message:
          "Use one row for a fixed group, at most 20 for repeatable groups.",
      },
    ),
  component: null,
});

const date = z.union([z.iso.date(), z.literal("")]);
export const DateRangeField = defineComponent({
  name: "DateRangeField",
  description:
    "Start/end calendar dates inside Form. value contains start and end as YYYY-MM-DD or blank, with local validation that end is not before start. Optional ranges may be entirely blank. This is not a booking.",
  props: z.object({
    name: identifier,
    label: shortText,
    value: z.object({ start: date, end: date }),
    required: z.boolean(),
  }),
  component: null,
});

export const RangeField = defineComponent({
  name: "RangeField",
  description:
    "Editable numeric low/high range inside Form with a stated unit, min/max/step and local validation. Publishes name_low and name_high for calculations. Both unknown values may be null; do not guess bounds for missing facts.",
  props: z
    .object({
      name: identifier,
      label: shortText,
      value: z.object({
        low: z.number().nullable(),
        high: z.number().nullable(),
      }),
      min: z.number(),
      max: z.number(),
      step: z.number().positive(),
      unit: z.string().max(50),
    })
    .refine(({ min, max }) => min < max, {
      message: "Minimum must be less than maximum.",
    })
    .refine(
      ({ value, min, max, step }) =>
        Object.values(value).every(
          (bound) =>
            bound === null ||
            (bound >= min &&
              bound <= max &&
              Math.abs(
                (bound - min) / step - Math.round((bound - min) / step),
              ) < 1e-8),
        ) &&
        (value.low === null || value.high === null || value.low <= value.high),
      {
        message: "Known bounds must be ordered and satisfy min, max and step.",
      },
    ),
  component: null,
});

export const DocumentEditor = defineComponent({
  name: "DocumentEditor",
  description:
    "Review and edit a text, Markdown or code draft inline inside Form, up to 20000 characters. Shows the original alongside edits and can download the current draft. value is initial current text; original is the review baseline. Submitting sends edited text to this chat, never sends email or writes an external document. Use native approval for external writes.",
  props: z.object({
    name: identifier,
    title: shortText,
    value: z.string().max(20_000),
    original: z.string().max(20_000),
    filename: z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,119}$/),
    required: z.boolean(),
  }),
  component: null,
});
