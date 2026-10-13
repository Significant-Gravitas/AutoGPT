import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import { identifier, shortText, sourceLink } from "./data-values";
import { artifactURI } from "./catalog-artifacts";

const webURL = z.url({ protocol: /^https?$/ }).max(2000);

export const resultOption = z.object({
  id: identifier,
  title: shortText,
  description: z.string().max(1000),
  facts: z
    .array(
      z.object({
        label: shortText,
        value: z.string().max(500),
        source: sourceLink.optional(),
      }),
    )
    .max(8),
  image: z.object({ url: artifactURI, alt: shortText }).optional(),
  source: sourceLink.optional(),
  amount: z.number().nullable().optional(),
});

export const ResultList = defineComponent({
  name: "ResultList",
  description:
    "Rich searchable option cards inside Form for 1–100 options, with fact-level sources and optional verified images {url,alt}; url may be the actual workspace file URI or HTTP(S) image output URL. Single or multiple selection submits stable ids at name; empty means undecided. Publishes name_amount (sum only when every selected amount is known, otherwise null) and name_count. unit explains amounts, e.g. USD/night; keep all amounts comparable. Preserve every candidate. Use Comparison for the user's two-option A/B swipe. This does not execute or approve actions.",
  props: z
    .object({
      name: identifier,
      label: shortText,
      options: z.array(resultOption).min(1).max(100),
      value: z.array(identifier).max(100),
      selection: z.enum(["single", "multiple"]),
      required: z.boolean(),
      unit: z.string().max(80),
    })
    .refine(
      ({ options, value, selection }) =>
        new Set(options.map((o) => o.id)).size === options.length &&
        new Set(value).size === value.length &&
        value.every((id) => options.some((o) => o.id === id)) &&
        (selection !== "single" || value.length <= 1),
      {
        message:
          "Use distinct option ids and existing selections; single selection permits at most one id.",
      },
    ),
  component: null,
});

export const Comparison = defineComponent({
  name: "Comparison",
  description:
    "Two selectable comparison cards inside Form. On mobile, swipe left chooses the first option and right the second; buttons and undo also work. Selection stays local until Form submission. value is an option id, or empty for no choice. Facts, images and source links must be supplied or verified. Preserve the scope of EACH option's supplied facts: a generic warranty duration cannot go under a part-specific coverage label, and labels need not match across cards. Use broad Warranty duration and Coverage unknown instead. Unknown condition for both does not prove either lower risk. Optional amount is a comparable numeric value in the SAME unit for both options; it is published as name_amount, with name_label for the title. Do not invent prices or use 0 for unknown amounts.",
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
