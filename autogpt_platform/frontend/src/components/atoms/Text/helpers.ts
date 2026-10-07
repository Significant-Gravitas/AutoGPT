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

export const variants = {
  // Headings
  h1: "font-poppins text-[2.75rem] font-semibold leading-14 tracking-[-0.033rem] text-foreground",
  h2: "font-poppins text-[2rem] font-medium leading-10 text-foreground tracking-[-0.02rem]",
  h3: "font-poppins text-[1.75rem] font-medium leading-10 text-foreground tracking-[-0.01313rem]",
  h4: "font-poppins text-[1.375rem] font-medium leading-6 text-foreground",
  h5: "font-poppins text-[1rem] font-medium leading-6 text-foreground",

  // Body Text
  lead: "font-sans text-[1.25rem] font-normal leading-7 text-foreground",
  "lead-medium":
    "font-sans text-[1.25rem] font-medium leading-7 text-foreground",
  "lead-semibold":
    "font-sans text-[1.25rem] font-semibold leading-7 text-foreground",
  large: "font-sans text-[1rem] font-normal leading-6.5 text-foreground",
  "large-medium":
    "font-sans text-[1rem] font-medium leading-6.5 text-foreground",
  "large-semibold":
    "font-sans text-[1rem] font-semibold leading-6.5 text-foreground",
  body: "font-sans text-[0.875rem] font-normal leading-5.5 text-foreground",
  "body-medium":
    "font-sans text-[0.875rem] font-medium leading-5.5 text-foreground",
  small: "font-sans text-[0.75rem] font-normal leading-4.5 text-foreground",
  "small-medium":
    "font-sans text-[0.75rem] font-medium leading-4.5 text-foreground",

  // Label Text
  label:
    "font-sans text-[0.6875rem] font-medium uppercase leading-5 tracking-[0.06875rem] text-foreground",
  eyebrow:
    "font-sans text-[0.75rem] font-medium uppercase leading-4 tracking-[0.06em] text-muted-foreground",
} as const;

export type Variant = keyof typeof variants;

/** Semantic colours. Omit for the variant's own colour (foreground). */
export const tones = {
  primary: "text-foreground",
  secondary: "text-zinc-700",
  muted: "text-muted-foreground",
  danger: "text-destructive",
  success: "text-success",
  warning: "text-warning",
  inherit: "text-inherit",
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
