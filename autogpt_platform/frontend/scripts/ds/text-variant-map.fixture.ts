/**
 * The Text atom's variants before they became roles plus props, with each
 * one's default element: copied from src/components/atoms/Text/helpers.ts at
 * 4745650dd7. text-variant-map.test.tsx renders every entry of
 * TEXT_VARIANT_MAP against this table. Never edit it.
 */
export const OLD_TEXT_VARIANTS: Record<
  string,
  { classes: string; element: string }
> = {
  hero: {
    classes:
      "font-poppins text-hero font-semibold tracking-title text-foreground-base",
    element: "h1",
  },
  h1: {
    classes:
      "font-poppins text-display font-semibold tracking-title text-foreground-base",
    element: "h1",
  },
  h2: {
    classes:
      "font-poppins text-title font-medium text-foreground-base tracking-title",
    element: "h2",
  },
  h3: {
    classes:
      "font-poppins text-heading font-medium text-foreground-base tracking-title",
    element: "h3",
  },
  "h3-tracking-tight": {
    classes:
      "font-poppins text-heading font-medium text-foreground-base tracking-tight",
    element: "h3",
  },
  h4: {
    classes: "font-poppins text-heading-sm font-medium text-foreground-base",
    element: "h4",
  },
  h5: {
    classes:
      "font-poppins text-body-lg font-medium leading-6 text-foreground-base",
    element: "h5",
  },
  "h5-semibold": {
    classes:
      "font-poppins text-body-lg font-semibold leading-6 text-foreground-base",
    element: "h5",
  },
  lead: {
    classes: "font-sans text-lead font-normal text-foreground-base",
    element: "p",
  },
  "lead-medium": {
    classes: "font-sans text-lead font-medium text-foreground-base",
    element: "p",
  },
  "lead-semibold": {
    classes: "font-sans text-lead font-semibold text-foreground-base",
    element: "p",
  },
  large: {
    classes: "font-sans text-body-lg font-normal text-foreground-base",
    element: "p",
  },
  "large-medium": {
    classes: "font-sans text-body-lg font-medium text-foreground-base",
    element: "p",
  },
  "large-semibold": {
    classes: "font-sans text-body-lg font-semibold text-foreground-base",
    element: "p",
  },
  "large-medium-tight": {
    classes:
      "font-sans text-body-lg font-medium leading-6 text-foreground-base",
    element: "p",
  },
  "large-semibold-tight": {
    classes:
      "font-sans text-body-lg font-semibold leading-6 text-foreground-base",
    element: "p",
  },
  "large-loose": {
    classes:
      "font-sans text-body-lg font-normal leading-8 text-foreground-base",
    element: "p",
  },
  body: {
    classes: "font-sans text-body font-normal text-foreground-base",
    element: "p",
  },
  "body-medium": {
    classes: "font-sans text-body font-medium text-foreground-base",
    element: "p",
  },
  "body-tight": {
    classes: "font-sans text-body font-normal leading-5 text-foreground-base",
    element: "p",
  },
  "body-medium-tight": {
    classes: "font-sans text-body font-medium leading-5 text-foreground-base",
    element: "p",
  },
  "body-md": {
    classes: "font-sans text-body-md font-normal text-foreground-base",
    element: "p",
  },
  "body-md-medium": {
    classes: "font-sans text-body-md font-medium text-foreground-base",
    element: "p",
  },
  "body-sm-medium-inherit": {
    classes:
      "font-sans text-body-sm font-medium leading-inherit text-foreground-base",
    element: "p",
  },
  small: {
    classes: "font-sans text-caption font-normal text-foreground-base",
    element: "p",
  },
  "small-medium": {
    classes: "font-sans text-caption font-medium text-foreground-base",
    element: "p",
  },
  "body-semibold-tight": {
    classes: "font-sans text-body font-semibold leading-5 text-foreground-base",
    element: "p",
  },
  "body-mono-tight": {
    classes: "font-mono text-body font-normal leading-5 text-foreground-base",
    element: "code",
  },
  "body-semibold": {
    classes: "font-sans text-body font-semibold text-foreground-base",
    element: "p",
  },
  "body-bold": {
    classes: "font-sans text-body font-bold text-foreground-base",
    element: "p",
  },
  "body-line-through": {
    classes:
      "font-sans text-body font-normal line-through text-foreground-base",
    element: "p",
  },
  "body-tight-italic": {
    classes:
      "font-sans text-body font-normal italic leading-5 text-foreground-base",
    element: "p",
  },
  "body-tight-touch": {
    classes:
      "font-sans text-body font-normal leading-5 text-foreground-base max-md:text-body-lg max-md:leading-6",
    element: "p",
  },
  "body-sm-medium-inherit-touch": {
    classes:
      "font-sans text-body-sm font-medium leading-inherit text-foreground-base max-md:text-body max-md:leading-5",
    element: "p",
  },
  "small-medium-tight-touch": {
    classes:
      "font-sans text-caption font-medium leading-4 text-foreground-base max-md:text-body max-md:leading-5",
    element: "p",
  },
  "small-mono": {
    classes: "font-mono text-caption font-normal text-foreground-base",
    element: "p",
  },
  "small-semibold": {
    classes: "font-sans text-caption font-semibold text-foreground-base",
    element: "p",
  },
  "small-semibold-tight": {
    classes:
      "font-sans text-caption font-semibold leading-4 text-foreground-base",
    element: "p",
  },
  "small-medium-lh-normal": {
    classes:
      "font-sans text-caption font-medium leading-normal text-foreground-base",
    element: "p",
  },
  "small-semibold-lh-normal": {
    classes:
      "font-sans text-caption font-semibold leading-normal text-foreground-base",
    element: "p",
  },
  "small-line-through": {
    classes:
      "font-sans text-caption font-normal line-through text-foreground-base",
    element: "p",
  },
  "small-tight": {
    classes:
      "font-sans text-caption font-normal leading-4 text-foreground-base",
    element: "p",
  },
  "small-tight-underline": {
    classes:
      "font-sans text-caption font-normal leading-4 underline text-foreground-base",
    element: "p",
  },
  "small-medium-tight": {
    classes:
      "font-sans text-caption font-medium leading-4 text-foreground-base",
    element: "p",
  },
  "small-uppercase-tight": {
    classes:
      "font-sans text-caption font-normal uppercase leading-4 text-foreground-base",
    element: "span",
  },
  "small-mono-tight": {
    classes:
      "font-mono text-caption font-normal leading-4 text-foreground-base",
    element: "code",
  },
  "small-mono-tight-tracking": {
    classes:
      "font-mono text-caption font-normal leading-4 tracking-tight text-foreground-base",
    element: "code",
  },
  "small-relaxed": {
    classes:
      "font-sans text-caption font-normal leading-relaxed text-foreground-base",
    element: "p",
  },
  "small-medium-mono-tight": {
    classes:
      "font-mono text-caption font-medium leading-4 text-foreground-base",
    element: "code",
  },
  "small-loose": {
    classes:
      "font-sans text-caption font-normal leading-5 text-foreground-base",
    element: "p",
  },
  "small-bold-loose": {
    classes: "font-sans text-caption font-bold leading-5 text-foreground-base",
    element: "p",
  },
  "small-medium-loose": {
    classes:
      "font-sans text-caption font-medium leading-5 text-foreground-base",
    element: "p",
  },
  "small-medium-loose-underline": {
    classes:
      "font-sans text-caption font-medium leading-5 underline text-foreground-base",
    element: "p",
  },
  "small-medium-inherit": {
    classes:
      "font-sans text-caption font-medium leading-inherit text-foreground-base",
    element: "span",
  },
  "tiny-inherit": {
    classes:
      "font-sans text-tiny font-normal leading-inherit text-foreground-base",
    element: "span",
  },
  label: {
    classes:
      "font-sans text-caption-sm font-medium uppercase leading-5 tracking-eyebrow text-foreground-base",
    element: "span",
  },
  eyebrow: {
    classes:
      "font-sans text-caption font-medium uppercase leading-4 tracking-eyebrow text-foreground-tertiary",
    element: "span",
  },
  "caption-sm-mono": {
    classes: "font-mono text-caption-sm font-normal text-foreground-base",
    element: "span",
  },
  "heading-md-bold": {
    classes: "font-sans text-heading-md font-bold text-foreground-base",
    element: "div",
  },
  "heading-md": {
    classes: "font-sans text-heading-md font-normal text-foreground-base",
    element: "span",
  },
  "body-medium-lh-none": {
    classes:
      "font-sans text-body font-medium leading-none text-foreground-base",
    element: "span",
  },
  "body-medium-lh-snug": {
    classes:
      "font-sans text-body font-medium leading-snug text-foreground-base",
    element: "p",
  },
  "small-lh-tight": {
    classes:
      "font-sans text-caption font-normal leading-tight text-foreground-base",
    element: "p",
  },
  "heading-md-bold-tracking-tight": {
    classes:
      "font-sans text-heading-md font-bold tracking-tight text-foreground-base",
    element: "h1",
  },
  "small-mono-bold-tight": {
    classes: "font-mono text-caption font-bold leading-4 text-foreground-base",
    element: "code",
  },
  "h3-semibold": {
    classes:
      "font-poppins text-heading font-semibold text-foreground-base tracking-title",
    element: "h3",
  },
  "caption-sm": {
    classes: "font-sans text-caption-sm font-normal text-foreground-base",
    element: "kbd",
  },
};
