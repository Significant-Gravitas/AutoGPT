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
  | "dd";

// Typography comes from the size roles in tailwind.config.ts; each bundles
// its line-height. TODO(tokens): SECRT-3152, the rem-based heading tracking
// and the `label` size/tracking have no role yet and stay raw.
export const variants = {
  // Headings
  h1: "font-poppins text-display font-semibold tracking-[-0.033rem] text-black",
  h2: "font-poppins text-title font-medium text-black tracking-[-0.02rem]",
  h3: "font-poppins text-heading font-medium text-black tracking-[-0.01313rem]",
  h4: "font-poppins text-heading-sm font-medium text-black",
  h5: "font-poppins text-body-lg font-medium leading-6 text-black",

  // Body Text
  lead: "font-sans text-lead font-normal text-black",
  "lead-medium": "font-sans text-lead font-medium text-black",
  "lead-semibold": "font-sans text-lead font-semibold text-black",
  large: "font-sans text-body-lg font-normal text-black",
  "large-medium": "font-sans text-body-lg font-medium text-black",
  "large-semibold": "font-sans text-body-lg font-semibold text-black",
  body: "font-sans text-body font-normal text-black",
  "body-medium": "font-sans text-body font-medium text-black",
  small: "font-sans text-caption font-normal text-black",
  "small-medium": "font-sans text-caption font-medium text-black",

  // Label Text
  label:
    "font-sans text-[0.6785rem] font-medium uppercase leading-5 tracking-[0.06785rem] text-black",
  eyebrow:
    "font-sans text-caption font-medium uppercase leading-4 tracking-eyebrow text-zinc-500",
} as const;

export type Variant = keyof typeof variants;

/** Semantic colours. Omit for the variant's own colour (black). */
export const tones = {
  primary: "text-zinc-900",
  secondary: "text-zinc-600",
  muted: "text-zinc-500",
  danger: "text-red-600",
} as const;

export type Tone = keyof typeof tones;

export const variantElementMap: Record<Variant, As> = {
  h1: "h1",
  h2: "h2",
  h3: "h3",
  h4: "h4",
  h5: "h5",
  lead: "p",
  "lead-medium": "p",
  "lead-semibold": "p",
  large: "p",
  "large-medium": "p",
  "large-semibold": "p",
  body: "p",
  "body-medium": "p",
  small: "p",
  "small-medium": "p",
  label: "span",
  eyebrow: "span",
};
