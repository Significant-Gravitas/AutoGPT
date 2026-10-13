import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import { identifier, shortText, sourceLink } from "./data-values";
import { artifactURI } from "./catalog-artifacts";

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

export const RankedList = defineComponent({
  name: "RankedList",
  description:
    "Rank a supplied list inside Form using accessible move-up/down buttons. Items have stable ids, labels and details. The submitted value is the ordered array of ids. No candidate is dropped and ranking is a preference, not execution.",
  props: z
    .object({
      name: identifier,
      label: shortText,
      items: z
        .array(
          z.object({
            id: identifier,
            label: shortText,
            detail: z.string().max(500),
          }),
        )
        .min(2)
        .max(30),
      value: z.array(identifier).min(2).max(30),
    })
    .refine(
      ({ items, value }) =>
        new Set(items.map((i) => i.id)).size === items.length &&
        new Set(value).size === items.length &&
        value.length === items.length &&
        value.every((id) => items.some((i) => i.id === id)),
      { message: "Ranking must contain every unique item id exactly once." },
    ),
  component: null,
});
