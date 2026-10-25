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
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-tight":
    "font-sans text-caption font-normal leading-4 text-foreground-base",
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

  // Label Text
  label:
    "font-sans text-[0.6785rem] font-medium uppercase leading-5 tracking-[0.06785rem] text-foreground-base",
  eyebrow:
    "font-sans text-caption font-medium uppercase leading-4 tracking-eyebrow text-foreground-tertiary",

  // Copy that moved off raw sizes in DS3a-2 (SECRT-3112), named like the
  // tight variants above: <size>[-<weight>][-mono]-<line height>.
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-semibold-tight":
    "font-sans text-body font-semibold leading-5 text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-mono-tight":
    "font-mono text-body font-normal leading-5 text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-semibold-tight":
    "font-sans text-caption font-semibold leading-4 text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1rem (text-xs); caption plus leading-4.
  "small-mono-bold-tight":
    "font-mono text-caption font-bold leading-4 text-foreground-base",
  // Keeps the parent's line height, like the raw size it replaced.
  "tiny-inherit":
    "font-sans text-tiny font-normal leading-inherit text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1.21875rem; caption plus leading-relaxed.
  "small-relaxed":
    "font-sans text-caption font-normal leading-relaxed text-foreground-base",
  // TODO(tokens): no mono role at caption-sm (0.6875rem/1rem, text-[11px]).
  "caption-sm-mono":
    "font-mono text-caption-sm font-normal text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1.25rem; caption plus leading-5.
  "small-loose":
    "font-sans text-caption font-normal leading-5 text-foreground-base",
  // TODO(tokens): no role is 0.75rem/1.25rem; caption plus leading-5.
  "small-medium-loose":
    "font-sans text-caption font-medium leading-5 text-foreground-base",
  "body-semibold": "font-sans text-body font-semibold text-foreground-base",
  // TODO(tokens): no role is 1rem/1.5rem (text-base); body-lg plus leading-6.
  "large-medium-tight":
    "font-sans text-body-lg font-medium leading-6 text-foreground-base",
  // TODO(tokens): SECRT-3152 no role is 1.125rem/1.75rem (text-lg); the raw size stays.
  "lg-semibold": "font-sans text-lg font-semibold text-foreground-base",
  // TODO(tokens): SECRT-3152 no role is 1.5rem/2rem (text-2xl); the raw size stays.
  "2xl-bold": "font-sans text-2xl font-bold text-foreground-base",
  // TODO(tokens): SECRT-3152 no role is 1.5rem/2rem (text-2xl); the raw size stays.
  "2xl": "font-sans text-2xl font-normal text-foreground-base",
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
  disabled: "text-foreground-disabled", // #C5C5C9
  // The parent's colour, for copy inside a control that changes it on hover.
  inherit: "text-inherit",
  // TODO(tokens): no role for text-emerald-300 #6EE7B7 (on a dark fill).
  "success-light": "text-emerald-300",
  // TODO(tokens): no role for text-emerald-700 #047857.
  "success-strong": "text-emerald-700",
  // TODO(tokens): no role for text-amber-800 #92400E.
  "warning-strong": "text-amber-800",
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
  "body-tight": "p",
  "body-medium-tight": "p",
  "body-md": "p",
  "body-md-medium": "p",
  "body-sm-medium-inherit": "p",
  small: "p",
  "small-medium": "p",
  "small-tight": "p",
  "small-medium-tight": "p",
  "small-uppercase-tight": "span",
  "small-mono-tight": "code",
  "small-mono-tight-tracking": "code",
  label: "span",
  eyebrow: "span",
  "body-semibold-tight": "p",
  "body-mono-tight": "code",
  "small-semibold-tight": "p",
  "small-mono-bold-tight": "code",
  "tiny-inherit": "span",
  "small-relaxed": "p",
  "caption-sm-mono": "span",
  "small-loose": "p",
  "small-medium-loose": "p",
  "body-semibold": "p",
  "large-medium-tight": "p",
  "lg-semibold": "h3",
  "2xl-bold": "div",
  "2xl": "span",
};
