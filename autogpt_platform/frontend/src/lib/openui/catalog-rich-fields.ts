import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";

export const TextAreaField = defineComponent({
  name: "TextAreaField",
  description:
    "Multiline text inside Form, up to 2000 characters. required=false makes it genuinely optional; use for notes, constraints or a brief.",
  props: z.object({
    name: z.string(),
    label: z.string(),
    value: z.string().max(2000),
    placeholder: z.string(),
    required: z.boolean(),
  }),
  component: null,
});

export const ToggleField = defineComponent({
  name: "ToggleField",
  description:
    "A labeled true/false checkbox inside Form. Use for a preference or constraint, never as consent to execute an external action.",
  props: z.object({ name: z.string(), label: z.string(), value: z.boolean() }),
  component: null,
});

export const MultiSelectField = defineComponent({
  name: "MultiSelectField",
  description:
    "A group of checkboxes inside Form. value is an array of selected option values; required=true requires at least one selection. Sends the actual array to the assistant.",
  props: z
    .object({
      name: z.string(),
      label: z.string(),
      value: z.array(z.string()).max(12),
      options: z
        .array(z.object({ label: z.string(), value: z.string().min(1) }))
        .min(1)
        .max(12),
      required: z.boolean(),
    })
    .refine(
      ({ value, options }) =>
        new Set(options.map((option) => option.value)).size ===
          options.length &&
        new Set(value).size === value.length &&
        value.every((item) => options.some((option) => option.value === item)),
      { message: "Use distinct options and selected values that match them." },
    ),
  component: null,
});
