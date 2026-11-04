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
// (`text-<step>`); each step bundles its line-height.
export const variants = {
  // Headings
  // The marketing hero: h1 at the scale's largest step.
  hero: "font-poppins text-hero font-semibold tracking-title text-foreground-base",
  h1: "font-poppins text-display font-semibold tracking-title text-foreground-base",
  h2: "font-poppins text-title font-medium text-foreground-base tracking-title",
  h3: "font-poppins text-heading font-medium text-foreground-base tracking-title",
  // TODO(tokens): no tracking token is -0.025em (tracking-tight); h3 with it.
  "h3-tracking-tight":
    "font-poppins text-heading font-medium text-foreground-base tracking-tight",
  h4: "font-poppins text-heading-sm font-medium text-foreground-base",
  h5: "font-poppins text-body-lg font-medium leading-6 text-foreground-base",
  // The h5 size and line-height at semibold (the wallet panel title).
  // TODO(tokens): no role is 1rem/1.5rem (text-base); body-lg plus leading-6.
  "h5-semibold":
    "font-poppins text-body-lg font-semibold leading-6 text-foreground-base",

  // Body Text
  lead: "font-sans text-lead font-normal text-foreground-base",
  "lead-medium": "font-sans text-lead font-medium text-foreground-base",
  "lead-semibold": "font-sans text-lead font-semibold text-foreground-base",
  large: "font-sans text-body-lg font-normal text-foreground-base",
  "large-medium": "font-sans text-body-lg font-medium text-foreground-base",
  "large-semibold": "font-sans text-body-lg font-semibold text-foreground-base",
  // TODO(tokens): no role is 1rem/1.5rem (text-base); body-lg plus leading-6.
  "large-medium-tight":
    "font-sans text-body-lg font-medium leading-6 text-foreground-base",
  // TODO(tokens): no role is 1rem/1.5rem (text-base); body-lg plus leading-6.
  "large-semibold-tight":
    "font-sans text-body-lg font-semibold leading-6 text-foreground-base",
  // TODO(tokens): no role is 1rem/2rem; body-lg plus leading-8.
  "large-loose":
    "font-sans text-body-lg font-normal leading-8 text-foreground-base",
  body: "font-sans text-body font-normal text-foreground-base",
  "body-medium": "font-sans text-body font-medium text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-tight":
    "font-sans text-body font-normal leading-5 text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-medium-tight":
    "font-sans text-body font-medium leading-5 text-foreground-base",
  "body-md": "font-sans text-body-md font-normal text-foreground-base",
  "body-md-medium": "font-sans text-body-md font-medium text-foreground-base",
  // Keeps the parent's line height, like the raw size it replaced.
  "body-sm-medium-inherit":
    "font-sans text-body-sm font-medium leading-inherit text-foreground-base",
  small: "font-sans text-caption font-normal text-foreground-base",
  "small-medium": "font-sans text-caption font-medium text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-semibold-tight":
    "font-sans text-body font-semibold leading-5 text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-mono-tight":
    "font-mono text-body font-normal leading-5 text-foreground-base",
  "body-semibold": "font-sans text-body font-semibold text-foreground-base",
  // A search match inside a body-medium title.
  "body-bold": "font-sans text-body font-bold text-foreground-base",
  // Struck through, e.g. a field whose connection is broken.
  "body-line-through":
    "font-sans text-body font-normal line-through text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-tight-italic":
    "font-sans text-body font-normal italic leading-5 text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5,
  // stepping up to body-lg (16px/24px) below md like the raw sizes it replaced.
  "body-tight-touch":
    "font-sans text-body font-normal leading-5 text-foreground-base max-md:text-body-lg max-md:leading-6",
  // body-sm-medium-inherit that steps up to body (14px/20px) below md, like
  // the responsive raw sizes it replaced.
  "body-sm-medium-inherit-touch":
    "font-sans text-body-sm font-medium leading-inherit text-foreground-base max-md:text-body max-md:leading-5",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4,
  // stepping up to body (14px/20px) below md like the raw sizes it replaced.
  "small-medium-tight-touch":
    "font-sans text-caption font-medium leading-4 text-foreground-base max-md:text-body max-md:leading-5",
  "small-mono": "font-mono text-caption font-normal text-foreground-base",
  "small-semibold": "font-sans text-caption font-semibold text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-semibold-tight":
    "font-sans text-caption font-semibold leading-4 text-foreground-base",
  // A tooltip title, at the tooltip's line height. TODO(tokens): no role is
  // 0.75rem with leading-normal.
  "small-medium-lh-normal":
    "font-sans text-caption font-medium leading-normal text-foreground-base",
  // TODO(tokens): no role is 0.75rem with leading-normal.
  "small-semibold-lh-normal":
    "font-sans text-caption font-semibold leading-normal text-foreground-base",
  // Struck through, e.g. a field whose connection is broken.
  "small-line-through":
    "font-sans text-caption font-normal line-through text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-tight":
    "font-sans text-caption font-normal leading-4 text-foreground-base",
  // A text link. TODO(tokens): no role is 0.75rem/1rem (text-xs).
  "small-tight-underline":
    "font-sans text-caption font-normal leading-4 underline text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-medium-tight":
    "font-sans text-caption font-medium leading-4 text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-uppercase-tight":
    "font-sans text-caption font-normal uppercase leading-4 text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-mono-tight":
    "font-mono text-caption font-normal leading-4 text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-mono-tight-tracking":
    "font-mono text-caption font-normal leading-4 tracking-tight text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1.21875rem; caption plus leading-relaxed.
  "small-relaxed":
    "font-sans text-caption font-normal leading-relaxed text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-medium-mono-tight":
    "font-mono text-caption font-medium leading-4 text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1.25rem; caption plus leading-5.
  "small-loose":
    "font-sans text-caption font-normal leading-5 text-foreground-base",
  // A search match inside a small-loose description. TODO(tokens): no role
  // is 0.75rem/1.25rem; caption plus leading-5.
  "small-bold-loose":
    "font-sans text-caption font-bold leading-5 text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1.25rem; caption plus leading-5.
  "small-medium-loose":
    "font-sans text-caption font-medium leading-5 text-foreground-base",
  // A link inside a block card. TODO(tokens): no role is 0.75rem/1.25rem.
  "small-medium-loose-underline":
    "font-sans text-caption font-medium leading-5 underline text-foreground-base",
  // Keeps the parent's line height, e.g. a link inside an Alert paragraph.
  "small-medium-inherit":
    "font-sans text-caption font-medium leading-inherit text-foreground-base",
  // Keeps the parent's line height, like the raw size it replaced.
  "tiny-inherit":
    "font-sans text-tiny font-normal leading-inherit text-foreground-base",

  // Label Text
  label:
    "font-sans text-caption-sm font-medium uppercase leading-5 tracking-eyebrow text-foreground-base",
  eyebrow:
    "font-sans text-caption font-medium uppercase leading-4 tracking-eyebrow text-foreground-tertiary",
  // TODO(tokens): no mono role at caption-sm (0.6875rem/1rem, text-[11px]).
  "caption-sm-mono":
    "font-mono text-caption-sm font-normal text-foreground-base",

  "heading-md-bold": "font-sans text-heading-md font-bold text-foreground-base",
  "heading-md": "font-sans text-heading-md font-normal text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1 (text-sm leading-none); body plus leading-none.
  "body-medium-lh-none":
    "font-sans text-body font-medium leading-none text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.375 (text-sm leading-snug); body plus leading-snug.
  "body-medium-lh-snug":
    "font-sans text-body font-medium leading-snug text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1.25 (text-xs leading-tight); caption plus leading-tight.
  "small-lh-tight":
    "font-sans text-caption font-normal leading-tight text-foreground-base",
  // TODO(tokens): no tracking token is -0.025em (tracking-tight).
  "heading-md-bold-tracking-tight":
    "font-sans text-heading-md font-bold tracking-tight text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-mono-bold-tight":
    "font-mono text-caption font-bold leading-4 text-foreground-base",
  // h3 at semibold.
  "h3-semibold":
    "font-poppins text-heading font-semibold text-foreground-base tracking-title",
  // The sans sibling of caption-sm-mono: caption-sm (0.6875rem/1rem) as it is.
  "caption-sm": "font-sans text-caption-sm font-normal text-foreground-base",
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
  "h3-tracking-tight": "h3",
  "h5-semibold": "h5",
  lead: "p",
  "lead-medium": "p",
  "lead-semibold": "p",
  large: "p",
  "large-medium": "p",
  "large-semibold": "p",
  "large-medium-tight": "p",
  "large-semibold-tight": "p",
  "large-loose": "p",
  body: "p",
  "body-medium": "p",
  "body-tight": "p",
  "body-semibold": "p",
  "body-bold": "p",
  "body-line-through": "p",
  "body-tight-italic": "p",
  "body-medium-tight": "p",
  "body-md": "p",
  "body-md-medium": "p",
  "body-tight-touch": "p",
  "body-semibold-tight": "p",
  "body-sm-medium-inherit": "p",
  "body-sm-medium-inherit-touch": "p",
  "small-medium-tight-touch": "p",
  small: "p",
  "small-medium": "p",
  "small-tight": "p",
  "small-tight-underline": "p",
  "small-semibold": "p",
  "small-semibold-tight": "p",
  "small-medium-lh-normal": "p",
  "small-semibold-lh-normal": "p",
  "small-line-through": "p",
  "small-medium-loose-underline": "p",
  "small-medium-tight": "p",
  "small-uppercase-tight": "span",
  "small-mono-tight": "code",
  "small-mono-tight-tracking": "code",
  "body-mono-tight": "code",
  "small-mono": "p",
  "small-relaxed": "p",
  "small-medium-mono-tight": "code",
  "small-loose": "p",
  "small-bold-loose": "p",
  "small-medium-loose": "p",
  "small-medium-inherit": "span",
  "tiny-inherit": "span",
  label: "span",
  eyebrow: "span",
  "caption-sm-mono": "span",
  "heading-md-bold": "div",
  "heading-md": "span",
  "body-medium-lh-none": "span",
  "body-medium-lh-snug": "p",
  "small-lh-tight": "p",
  "heading-md-bold-tracking-tight": "h1",
  "small-mono-bold-tight": "code",
  "h3-semibold": "h3",
  "caption-sm": "kbd",
};
