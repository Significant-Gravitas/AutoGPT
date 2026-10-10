import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import { Comparison, CostTable } from "./catalog-connected";
import {
  TextAreaField,
  ToggleField,
  MultiSelectField,
} from "./catalog-rich-fields";

export const SelectField = defineComponent({
  name: "SelectField",
  description:
    "A labeled dropdown with one to 12 options inside a Form. The initial value must match an option value. Use unique nonempty option values.",
  props: z
    .object({
      name: z.string(),
      label: z.string(),
      value: z.string(),
      options: z
        .array(z.object({ label: z.string(), value: z.string().min(1) }))
        .min(1)
        .max(12),
    })
    .refine(
      ({ value, options }) =>
        options.some((option) => option.value === value) &&
        new Set(options.map((option) => option.value)).size === options.length,
    ),
  component: null,
});

export const DateField = defineComponent({
  name: "DateField",
  description:
    "A calendar date input inside a Form. Use YYYY-MM-DD, or an empty string when the user should choose. This does not schedule anything.",
  props: z.object({
    name: z.string(),
    label: z.string(),
    value: z.union([z.iso.date(), z.literal("")]),
  }),
  component: null,
});

export const NumberField = defineComponent({
  name: "NumberField",
  description:
    "A numeric input inside a Form. Provide a numeric initial value, minimum/maximum (null if unbounded), and a positive step. The initial value must satisfy those constraints.",
  props: z
    .object({
      name: z.string(),
      label: z.string(),
      value: z.number(),
      min: z.number().nullable().optional(),
      max: z.number().nullable().optional(),
      step: z.number().positive(),
    })
    .refine(({ value, min, max, step }) => {
      if (min != null && value < min) return false;
      if (max != null && value > max) return false;
      if (min == null) return true;
      const steps = (value - min) / step;
      return Math.abs(steps - Math.round(steps)) < 1e-8;
    }),
  component: null,
});

export const Field = defineComponent({
  name: "Field",
  description:
    "An editable, labeled text field inside a Form. Use a unique name.",
  props: z.object({
    name: z.string(),
    label: z.string(),
    value: z.string(),
    placeholder: z.string(),
    required: z.boolean().optional(),
  }),
  component: null,
});

export const Form = defineComponent({
  name: "Form",
  description:
    "Collect a brief with at most six fields. Submitting sends the edited values to the assistant; no external action is executed.",
  props: z.object({
    name: z.string(),
    title: z.string(),
    fields: z
      .array(
        z.union([
          Field.ref,
          SelectField.ref,
          DateField.ref,
          NumberField.ref,
          Comparison.ref,
          CostTable.ref,
          TextAreaField.ref,
          ToggleField.ref,
          MultiSelectField.ref,
        ]),
      )
      .max(6),
    submitLabel: z.string(),
    message: z.string(),
  }),
  component: null,
});
