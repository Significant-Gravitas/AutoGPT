/**
 * The Text atom's 67 old variant names and what each is now: a role
 * (src/components/atoms/Text/helpers.ts) plus the props that add what the old
 * name baked in, and the `as` that keeps the old default element where the
 * role's differs. scripts/ds/text-variants.ts rewrites call sites from this
 * table, and scripts/ds/text-variant-map.test.tsx proves every entry renders
 * the old variant's exact class set and element.
 */
import type { As, Variant } from "../../src/components/atoms/Text/helpers";
import type { TextProps } from "../../src/components/atoms/Text/Text";

export type TextVariantProps = Pick<
  TextProps,
  | "weight"
  | "mono"
  | "italic"
  | "decoration"
  | "transform"
  | "tracking"
  | "leading"
  | "narrowSize"
>;

export interface TextVariantMapping {
  variant: Variant;
  props?: TextVariantProps;
  /** The old default element, where the role's own default differs. */
  as?: As;
}

export const TEXT_VARIANT_MAP = {
  hero: { variant: "hero" },
  h1: { variant: "h1" },
  h2: { variant: "h2" },
  h3: { variant: "h3" },
  "h3-tracking-tight": { variant: "h3", props: { tracking: "tight" } },
  "h3-semibold": { variant: "h3", props: { weight: "semibold" } },
  h4: { variant: "h4" },
  h5: { variant: "h5" },
  "h5-semibold": { variant: "h5", props: { weight: "semibold" } },

  "heading-md": { variant: "heading-md" },
  "heading-md-bold": {
    variant: "heading-md",
    props: { weight: "bold" },
    as: "div",
  },
  "heading-md-bold-tracking-tight": {
    variant: "heading-md",
    props: { weight: "bold", tracking: "tight" },
    as: "h1",
  },

  lead: { variant: "lead" },
  "lead-medium": { variant: "lead", props: { weight: "medium" } },
  "lead-semibold": { variant: "lead", props: { weight: "semibold" } },

  large: { variant: "large" },
  "large-medium": { variant: "large", props: { weight: "medium" } },
  "large-semibold": { variant: "large", props: { weight: "semibold" } },
  "large-medium-tight": {
    variant: "large",
    props: { weight: "medium", leading: "6" },
  },
  "large-semibold-tight": {
    variant: "large",
    props: { weight: "semibold", leading: "6" },
  },
  "large-loose": { variant: "large", props: { leading: "8" } },

  "body-md": { variant: "body-md" },
  "body-md-medium": { variant: "body-md", props: { weight: "medium" } },

  body: { variant: "body" },
  "body-medium": { variant: "body", props: { weight: "medium" } },
  "body-semibold": { variant: "body", props: { weight: "semibold" } },
  "body-bold": { variant: "body", props: { weight: "bold" } },
  "body-tight": { variant: "body", props: { leading: "5" } },
  "body-medium-tight": {
    variant: "body",
    props: { weight: "medium", leading: "5" },
  },
  "body-semibold-tight": {
    variant: "body",
    props: { weight: "semibold", leading: "5" },
  },
  "body-mono-tight": {
    variant: "body",
    props: { mono: true, leading: "5" },
    as: "code",
  },
  "body-tight-italic": {
    variant: "body",
    props: { italic: true, leading: "5" },
  },
  "body-tight-touch": {
    variant: "body",
    props: { leading: "5", narrowSize: "large" },
  },
  "body-line-through": {
    variant: "body",
    props: { decoration: "line-through" },
  },
  "body-medium-lh-none": {
    variant: "body",
    props: { weight: "medium", leading: "none" },
    as: "span",
  },
  "body-medium-lh-snug": {
    variant: "body",
    props: { weight: "medium", leading: "snug" },
  },

  "body-sm-medium-inherit": {
    variant: "body-sm",
    props: { weight: "medium", leading: "inherit" },
  },
  "body-sm-medium-inherit-touch": {
    variant: "body-sm",
    props: { weight: "medium", leading: "inherit", narrowSize: "body" },
  },

  small: { variant: "small" },
  "small-medium": { variant: "small", props: { weight: "medium" } },
  "small-semibold": { variant: "small", props: { weight: "semibold" } },
  "small-mono": { variant: "small", props: { mono: true } },
  "small-line-through": {
    variant: "small",
    props: { decoration: "line-through" },
  },
  "small-tight": { variant: "small", props: { leading: "4" } },
  "small-tight-underline": {
    variant: "small",
    props: { leading: "4", decoration: "underline" },
  },
  "small-medium-tight": {
    variant: "small",
    props: { weight: "medium", leading: "4" },
  },
  "small-semibold-tight": {
    variant: "small",
    props: { weight: "semibold", leading: "4" },
  },
  "small-uppercase-tight": {
    variant: "small",
    props: { transform: "uppercase", leading: "4" },
    as: "span",
  },
  "small-mono-tight": {
    variant: "small",
    props: { mono: true, leading: "4" },
    as: "code",
  },
  "small-mono-tight-tracking": {
    variant: "small",
    props: { mono: true, leading: "4", tracking: "tight" },
    as: "code",
  },
  "small-medium-mono-tight": {
    variant: "small",
    props: { weight: "medium", mono: true, leading: "4" },
    as: "code",
  },
  "small-mono-bold-tight": {
    variant: "small",
    props: { weight: "bold", mono: true, leading: "4" },
    as: "code",
  },
  "small-medium-tight-touch": {
    variant: "small",
    props: { weight: "medium", leading: "4", narrowSize: "body" },
  },
  "small-loose": { variant: "small", props: { leading: "5" } },
  "small-bold-loose": {
    variant: "small",
    props: { weight: "bold", leading: "5" },
  },
  "small-medium-loose": {
    variant: "small",
    props: { weight: "medium", leading: "5" },
  },
  "small-medium-loose-underline": {
    variant: "small",
    props: { weight: "medium", leading: "5", decoration: "underline" },
  },
  "small-relaxed": { variant: "small", props: { leading: "relaxed" } },
  "small-lh-tight": { variant: "small", props: { leading: "tight" } },
  "small-medium-lh-normal": {
    variant: "small",
    props: { weight: "medium", leading: "normal" },
  },
  "small-semibold-lh-normal": {
    variant: "small",
    props: { weight: "semibold", leading: "normal" },
  },
  "small-medium-inherit": {
    variant: "small",
    props: { weight: "medium", leading: "inherit" },
    as: "span",
  },

  "caption-sm": { variant: "caption-sm", as: "kbd" },
  "caption-sm-mono": { variant: "caption-sm", props: { mono: true } },
  "tiny-inherit": { variant: "tiny", props: { leading: "inherit" } },

  label: { variant: "label" },
  eyebrow: { variant: "eyebrow" },
} satisfies Record<string, TextVariantMapping>;

export type OldTextVariant = keyof typeof TEXT_VARIANT_MAP;

export function isOldTextVariant(name: string): name is OldTextVariant {
  return Object.hasOwn(TEXT_VARIANT_MAP, name);
}
