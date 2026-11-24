import { cva } from "class-variance-authority";
import { tones } from "@/components/styles/tones";

export type As =
  | "h1"
  | "h2"
  | "h3"
  | "h4"
  | "h5"
  | "h6"
  | "p"
  | "span"
  | "div"
  | "code"
  | "label"
  | "kbd"
  | "li"
  | "dt"
  | "dd"
  | "th"
  | "td"
  | "pre"
  | "button";

// Typography comes from the type scale in src/app/tokens/typography.css
// (`text-<step>`); each step bundles its line-height. A variant names a type
// role only: its size, family and the weight, tracking or line height the role
// always has. Everything else is a prop (`textModifiers` below).
export const variants = {
  // Headings
  // The marketing hero: h1 at the scale's largest step.
  hero: "font-poppins text-hero font-semibold tracking-title text-foreground-base",
  h1: "font-poppins text-display font-semibold tracking-title text-foreground-base",
  h2: "font-poppins text-title font-medium text-foreground-base tracking-title",
  h3: "font-poppins text-heading font-medium text-foreground-base tracking-title",
  h4: "font-poppins text-heading-sm font-medium text-foreground-base",
  // TODO(tokens): no role is 1rem/1.5rem (text-base); body-lg plus leading-6.
  h5: "font-poppins text-body-lg font-medium leading-6 text-foreground-base",
  "heading-md": "font-sans text-heading-md font-normal text-foreground-base",

  // Body Text
  lead: "font-sans text-lead font-normal text-foreground-base",
  large: "font-sans text-body-lg font-normal text-foreground-base",
  "body-md": "font-sans text-body-md font-normal text-foreground-base",
  body: "font-sans text-body font-normal text-foreground-base",
  "body-sm": "font-sans text-body-sm font-normal text-foreground-base",
  small: "font-sans text-caption font-normal text-foreground-base",
  "caption-sm": "font-sans text-caption-sm font-normal text-foreground-base",
  tiny: "font-sans text-tiny font-normal text-foreground-base",

  // Label Text
  label:
    "font-sans text-caption-sm font-medium uppercase leading-5 tracking-eyebrow text-foreground-base",
  eyebrow:
    "font-sans text-caption font-medium uppercase leading-4 tracking-eyebrow text-foreground-tertiary",
} as const;

export type Variant = keyof typeof variants;

// The props that adjust a variant (Text.tsx). Each applies after the
// variant's classes, so `weight="semibold"` replaces its `font-medium`.
export const textModifiers = cva("", {
  variants: {
    tone: tones,
    weight: {
      normal: "font-normal",
      medium: "font-medium",
      semibold: "font-semibold",
      bold: "font-bold",
    },
    mono: { true: "font-mono" },
    tabular: { true: "tabular-nums" },
    italic: { true: "italic" },
    decoration: {
      underline: "underline",
      "line-through": "line-through",
      none: "no-underline",
    },
    transform: {
      uppercase: "uppercase",
      lowercase: "lowercase",
      capitalize: "capitalize",
      none: "normal-case",
    },
    tracking: {
      tight: "tracking-tight",
      normal: "tracking-normal",
      wide: "tracking-wide",
      widest: "tracking-widest",
      title: "tracking-title",
      heading: "tracking-heading",
      display: "tracking-display",
      caps: "tracking-caps",
      eyebrow: "tracking-eyebrow",
    },
    // TODO(tokens): the fixed steps (4, 5, 6, 8) are Tailwind's line heights
    // for the raw sizes the scale replaced; no role bundles them.
    leading: {
      none: "leading-none",
      tight: "leading-tight",
      snug: "leading-snug",
      normal: "leading-normal",
      relaxed: "leading-relaxed",
      inherit: "leading-inherit",
      "4": "leading-4",
      "4.5": "leading-4.5",
      "5": "leading-5",
      "6": "leading-6",
      "7": "leading-7",
      "8": "leading-8",
      "9": "leading-9",
      "10": "leading-10",
    },
    // Below md, a step up to body (14/20) or large (16/24), like the
    // responsive raw sizes it replaced.
    narrowSize: {
      body: "max-md:text-body max-md:leading-5",
      large: "max-md:text-body-lg max-md:leading-6",
    },
    // Animates a colour change, e.g. a hover colour set by a parent.
    motion: { colors: "transition-colors" },
    align: {
      left: "text-left",
      center: "text-center",
      right: "text-right",
      start: "text-start",
      end: "text-end",
    },
  },
});

export const variantElementMap: Record<Variant, As> = {
  hero: "h1",
  h1: "h1",
  h2: "h2",
  h3: "h3",
  h4: "h4",
  h5: "h5",
  "heading-md": "span",
  lead: "p",
  large: "p",
  "body-md": "p",
  body: "p",
  "body-sm": "p",
  small: "p",
  "caption-sm": "span",
  tiny: "span",
  label: "span",
  eyebrow: "span",
};
