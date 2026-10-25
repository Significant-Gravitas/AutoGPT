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
  // TODO(tokens): no tracking token is -0.025em (tracking-tight); h3 with it.
  "h3-tracking-tight":
    "font-poppins text-heading font-medium text-foreground-base tracking-tight",
  h4: "font-poppins text-heading-sm font-medium text-foreground-base",
  h5: "font-poppins text-body-lg font-medium leading-6 text-foreground-base",

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
  body: "font-sans text-body font-normal text-foreground-base",
  "body-medium": "font-sans text-body font-medium text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-tight":
    "font-sans text-body font-normal leading-5 text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-medium-tight":
    "font-sans text-body font-medium leading-5 text-foreground-base",
  "body-semibold": "font-sans text-body font-semibold text-foreground-base",
  // A search match inside a body-medium title.
  "body-bold": "font-sans text-body font-bold text-foreground-base",
  // Struck through, e.g. a field whose connection is broken.
  "body-line-through":
    "font-sans text-body font-normal line-through text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-tight-italic":
    "font-sans text-body font-normal italic leading-5 text-foreground-base",
  // TODO(tokens): no role is 0.875rem/1.25rem (text-sm); body plus leading-5.
  "body-semibold-tight":
    "font-sans text-body font-semibold leading-5 text-foreground-base",
  small: "font-sans text-caption font-normal text-foreground-base",
  "small-medium": "font-sans text-caption font-medium text-foreground-base",
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
  "small-mono-tight":
    "font-mono text-caption font-normal leading-4 text-foreground-base",
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
  // TODO(tokens): no role is 0.75rem/1.21875rem; caption plus leading-relaxed.
  "small-relaxed":
    "font-sans text-caption font-normal leading-relaxed text-foreground-base",
  // Keeps the parent's line height, e.g. a link inside an Alert paragraph.
  "small-medium-inherit":
    "font-sans text-caption font-medium leading-inherit text-foreground-base",
  // Keeps the parent's line height, like the raw size it replaced.
  "tiny-inherit":
    "font-sans text-tiny font-normal leading-inherit text-foreground-base",
  // Keeps the parent's line height, like the raw size it replaced.
  // TODO(tokens): no mono role at caption-sm (0.6875rem, text-[11px]).
  "caption-sm-mono-inherit":
    "font-mono text-caption-sm font-normal leading-inherit text-foreground-base",

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
  // The parent's colour, for copy inside a control that changes it on hover.
  inherit: "text-inherit",
} as const;

export type Tone = keyof typeof tones;

export const variantElementMap: Record<Variant, As> = {
  h1: "h1",
  h2: "h2",
  h3: "h3",
  h4: "h4",
  h5: "h5",
  "h3-tracking-tight": "h3",
  lead: "p",
  "lead-medium": "p",
  "lead-semibold": "p",
  large: "p",
  "large-medium": "p",
  "large-semibold": "p",
  "large-medium-tight": "p",
  "large-semibold-tight": "p",
  body: "p",
  "body-medium": "p",
  "body-tight": "p",
  "body-semibold": "p",
  "body-bold": "p",
  "body-line-through": "p",
  "body-tight-italic": "p",
  "body-medium-tight": "p",
  "body-semibold-tight": "p",
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
  "small-mono-tight": "code",
  "small-medium-mono-tight": "code",
  "small-loose": "p",
  "small-bold-loose": "p",
  "small-medium-loose": "p",
  "small-relaxed": "p",
  "small-medium-inherit": "span",
  "tiny-inherit": "span",
  "caption-sm-mono-inherit": "span",
  label: "span",
  eyebrow: "span",
};
