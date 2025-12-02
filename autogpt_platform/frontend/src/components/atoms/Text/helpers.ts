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
  | "kbd";

export const variants = {
  // Headings
  h1: "font-poppins text-[2.75rem] font-[600] leading-[3.5rem] tracking-[-0.033rem] text-zinc-800",
  h2: "font-poppins text-[2rem] font-[500] leading-[2.5rem] text-zinc-800 tracking-[-0.02rem]",
  h3: "font-poppins text-[1.75rem] font-[500] leading-[2.5rem] text-zinc-800 tracking-[-0.01313rem]",
  h4: "font-poppins text-[1.375rem] font-[500] leading-[1.5rem] text-zinc-800",
  h5: "font-poppins text-[1rem] font-[500] leading-[1.5rem] text-zinc-800",

  // Body Text
  lead: "font-sans text-[1.25rem] font-[400] leading-[1.75rem] text-zinc-800",
  "lead-medium":
    "font-sans text-[1.25rem] font-[500] leading-[1.75rem] text-zinc-800",
  "lead-semibold":
    "font-sans text-[1.25rem] font-[600] leading-[1.75rem] text-zinc-800",
  large: "font-sans text-[1rem] font-[400] leading-[1.625rem] text-zinc-800",
  "large-medium":
    "font-sans text-[1rem] font-[500] leading-[1.625rem] text-zinc-800",
  "large-semibold":
    "font-sans text-[1rem] font-[600] leading-[1.625rem] text-zinc-800",
  body: "font-sans text-[0.875rem] font-[400] leading-[1.375rem] text-zinc-800",
  "body-medium":
    "font-sans text-[0.875rem] font-[500] leading-[1.375rem] text-zinc-800",
  small: "font-sans text-[0.75rem] font-[400] leading-[1.125rem] text-zinc-800",
  "small-medium":
    "font-sans text-[0.75rem] font-[500] leading-[1.125rem] text-zinc-800",

  // Label Text
  label:
    "font-sans text-[0.6785rem] font-medium uppercase leading-[1.25rem] tracking-[0.06785rem] text-zinc-800",
} as const;

export type Variant = keyof typeof variants;

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
};
