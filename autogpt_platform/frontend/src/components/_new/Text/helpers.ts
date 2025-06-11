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
  h1: "font-poppins text-5xl font-semibold leading-[56px] text-zinc-800",
  h2: "font-poppins text-4xl font-normal leading-[52px] text-zinc-800",
  h3: "font-poppins text-3xl font-medium leading-10 text-zinc-800",
  h4: "font-poppins text-base font-medium leading-normal text-zinc-800",

  // Body Text
  lead: "font-sans text-xl font-normal leading-loose text-muted-zinc-800",
  large: "font-sans text-base font-normal leading-normal text-zinc-800",
  "large-medium":
    "font-sans text-base font-medium leading-normal text-zinc-800",
  "large-semibold":
    "font-sans text-base font-semibold leading-normal text-zinc-800",
  body: "font-sans text-sm font-normal leading-snug text-zinc-800",
  "body-medium": "font-sans text-sm font-medium leading-snug text-zinc-800",
  small: "font-sans text-xs font-normal leading-tight text-zinc-800",
  "small-medium": "font-sans text-xs font-medium leading-tight text-zinc-800",
  subtle:
    "font-sans text-xs font-medium uppercase leading-tight tracking-wide text-zinc-800",
} as const;

export type Variant = keyof typeof variants;

export const variantElementMap: Record<Variant, As> = {
  h1: "h1",
  h2: "h2",
  h3: "h3",
  h4: "h4",
  lead: "p",
  large: "p",
  "large-medium": "p",
  "large-semibold": "p",
  body: "p",
  "body-medium": "p",
  small: "p",
  "small-medium": "p",
  subtle: "p",
};
