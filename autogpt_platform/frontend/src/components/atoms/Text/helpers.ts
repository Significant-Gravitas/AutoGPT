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
  h1: "font-poppins text-display font-semibold tracking-[-0.033rem] text-foreground-base",
  h2: "font-poppins text-title font-medium text-foreground-base tracking-[-0.02rem]",
  h3: "font-poppins text-heading font-medium text-foreground-base tracking-[-0.01313rem]",
  h4: "font-poppins text-heading-sm font-medium text-foreground-base",
  h5: "font-poppins text-body-lg font-medium leading-6 text-foreground-base",
  // The h5 size and line-height at semibold (the wallet panel title).
  "h5-semibold":
    "font-poppins text-body-lg font-semibold leading-6 text-foreground-base",

  // Body Text
  lead: "font-sans text-lead font-normal text-foreground-base",
  "lead-medium": "font-sans text-lead font-medium text-foreground-base",
  "lead-semibold": "font-sans text-lead font-semibold text-foreground-base",
  large: "font-sans text-body-lg font-normal text-foreground-base",
  "large-medium": "font-sans text-body-lg font-medium text-foreground-base",
  "large-semibold": "font-sans text-body-lg font-semibold text-foreground-base",
  body: "font-sans text-body font-normal text-foreground-base",
  "body-medium": "font-sans text-body font-medium text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-tight":
    "font-sans text-body font-normal leading-5 text-foreground-base",
  small: "font-sans text-caption font-normal text-foreground-base",
  "small-medium": "font-sans text-caption font-medium text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-tight":
    "font-sans text-caption font-normal leading-4 text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1.25rem; caption plus leading-5.
  "small-loose":
    "font-sans text-caption font-normal leading-5 text-foreground-base",

  // Label Text
  label:
    "font-sans text-[0.6785rem] font-medium uppercase leading-5 tracking-[0.06785rem] text-foreground-base",
  eyebrow:
    "font-sans text-caption font-medium uppercase leading-4 tracking-eyebrow text-foreground-tertiary",
} as const;

export type Variant = keyof typeof variants;

/**
 * Semantic colours, each a colour role at the hex it rendered before the
 * roles existed. Omit for the variant's own colour (foreground-base #141414).
 */
export const tones = {
  primary: "text-foreground-strong", // #2C2C30
  emphasis: "text-foreground-emphasis", // #3E3E43
  body: "text-foreground-body", // #505057
  secondary: "text-foreground-secondary", // #68686F
  muted: "text-foreground-tertiary", // #83838C
  subtle: "text-foreground-subtle", // #ADADB3
  danger: "text-error-hover", // #D93636
  error: "text-error", // #EF4444
  warning: "text-warning-foreground", // #B45309
  success: "text-success", // #149443
} as const;

export type Tone = keyof typeof tones;

export const variantElementMap: Record<Variant, As> = {
  h1: "h1",
  h2: "h2",
  h3: "h3",
  h4: "h4",
  h5: "h5",
  "h5-semibold": "h5",
  lead: "p",
  "lead-medium": "p",
  "lead-semibold": "p",
  large: "p",
  "large-medium": "p",
  "large-semibold": "p",
  body: "p",
  "body-medium": "p",
  "body-tight": "p",
  small: "p",
  "small-medium": "p",
  "small-tight": "p",
  "small-loose": "p",
  label: "span",
  eyebrow: "span",
};
