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
  h1: "font-poppins text-[2.75rem] font-[600] leading-[3.5rem] tracking-[-0.033rem] text-black",
  h2: "font-poppins text-[2rem] font-[500] leading-[2.5rem] text-black tracking-[-0.02rem]",
  h3: "font-poppins text-[1.75rem] font-[500] leading-[2.5rem] text-black tracking-[-0.01313rem]",
  h4: "font-poppins text-[1.375rem] font-[500] leading-[1.5rem] text-black",
  h5: "font-poppins text-[1rem] font-[500] leading-[1.5rem] text-black",

  // Body Text
  lead: "font-sans text-[1.25rem] font-[400] leading-[1.75rem] text-black",
  "lead-medium":
    "font-sans text-[1.25rem] font-[500] leading-[1.75rem] text-black",
  "lead-semibold":
    "font-sans text-[1.25rem] font-[600] leading-[1.75rem] text-black",
  large: "font-sans text-[1rem] font-[400] leading-[1.625rem] text-black",
  "large-medium":
    "font-sans text-[1rem] font-[500] leading-[1.625rem] text-black",
  "large-semibold":
    "font-sans text-[1rem] font-[600] leading-[1.625rem] text-black",
  body: "font-sans text-[0.875rem] font-[400] leading-[1.375rem] text-black",
  "body-medium":
    "font-sans text-[0.875rem] font-[500] leading-[1.375rem] text-black",
  small: "font-sans text-[0.75rem] font-[400] leading-[1.125rem] text-black",
  "small-medium":
    "font-sans text-[0.75rem] font-[500] leading-[1.125rem] text-black",

  // Label Text
  label:
    "font-sans text-[0.6785rem] font-medium uppercase leading-[1.25rem] tracking-[0.06785rem] text-black",
  eyebrow:
    "font-sans text-[0.75rem] font-[500] uppercase leading-[1rem] tracking-[0.06em] text-zinc-500",
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
