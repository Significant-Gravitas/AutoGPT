import {
  linkBaseClasses,
  linkFocusClasses,
  linkVariantClasses,
} from "@/components/atoms/Link/Link";
import { cn } from "@/lib/utils";
import { tones } from "@/components/styles/tones";
import { cva, VariantProps } from "class-variance-authority";
import { IconSvgElement } from "@hugeicons/react";
import { LinkProps } from "next/link";

// A ghost Button's box: no fill or border until hover. A ghost `tone` (below)
// gives it a quieter text colour.
const GHOST =
  "bg-transparent border-transparent hover:bg-surface-muted hover:border-surface-muted disabled:text-surface-active";

// Text colour over the variant's own: the shared tones (see
// src/components/styles/tones.ts), and four for a ghost Button that also
// change on hover.
const BUTTON_TONES = {
  ...tones,
  // A secondary action in a row or card: grey until hovered.
  quiet: "text-foreground-tertiary hover:text-foreground-body",
  "quiet-subtle": "text-foreground-subtle hover:text-foreground-body",
  // A remove or delete action that turns red on hover.
  "quiet-danger": "text-foreground-tertiary hover:text-error",
  "quiet-warning":
    "text-warning-foreground hover:bg-warning-surface hover:border-warning-surface",
} as const;

// Corner radius, from the radius tokens. A look that squares controls
// changes every Button through these.
const BUTTON_SHAPES = {
  pill: "rounded-pill",
  rounded: "rounded-xl",
  soft: "rounded-lg",
  sharp: "rounded-md",
} as const;

// The shape a variant takes when the call site sets none: `pill`, except the
// toggle's and the floating button's squarer corner. A link has no box, so
// no shape (it keeps the Link atom's `rounded-sm` focus ring).
export function defaultButtonShape(variant: string | null | undefined) {
  if (variant === "link") return null;
  if (variant === "toggle" || variant === "floating") return "sharp";
  return undefined;
}

// Extended button variants based on our design system. The base sets no
// disabled opacity: a variant that fades while disabled sets its own.
// TODO(tokens): no spacing step is min-w-[7.7rem], min-w-[5.5rem] or
// h-[2.875rem], so they stay raw.
export const extendedButtonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus disabled:pointer-events-none font-sans leading-snug border min-w-[7.7rem]",
  {
    variants: {
      variant: {
        primary:
          "bg-action border-action text-action-foreground hover:bg-action-hover hover:border-action-hover disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base",
        secondary:
          "bg-surface border-border-base text-foreground-emphasis shadow-sm hover:bg-surface-muted hover:border-border-strong disabled:text-foreground-disabled disabled:bg-surface-muted disabled:border-separator disabled:shadow-none",
        destructive:
          "bg-error border-error text-action-foreground hover:bg-error-hover hover:border-error-hover disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base",
        outline:
          "bg-transparent border-border-strong text-foreground-base hover:bg-surface-hover hover:border-border-strong disabled:border-border-base disabled:text-surface-active",
        ghost: `${GHOST} text-foreground-base`,
        // An upgrade or promotion call to action.
        brand:
          "bg-brand border-brand text-action-foreground hover:bg-accent-strong hover:border-accent-strong disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base disabled:opacity-100",
        // Goes ahead despite a warning, e.g. an update that breaks links.
        warning:
          "bg-warning-foreground border-warning-foreground text-action-foreground hover:bg-warning-foreground/90 hover:border-warning-foreground disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base disabled:opacity-100",
        // A secondary Button showing that its action worked, e.g. "Copied".
        success:
          "bg-success/30 border-success text-foreground-base shadow-sm hover:bg-success/40 hover:border-success disabled:shadow-none disabled:opacity-100",
        // An icon action in a toolbar such as the chat composer: no box
        // until hovered, faded while disabled.
        tray: "bg-transparent border-transparent text-foreground-base hover:bg-surface-hover hover:border-transparent disabled:opacity-40 !min-w-0",
        icon: "bg-surface text-foreground-emphasis border border-border-base shadow-sm hover:bg-surface-muted hover:border-border-strong disabled:shadow-none !min-w-0",
        toggle:
          "bg-transparent border-transparent text-foreground-tertiary hover:bg-transparent hover:border-transparent hover:text-foreground-emphasis aria-pressed:bg-surface-hover aria-pressed:text-foreground-strong disabled:opacity-50",
        // An icon action floating over a cover image.
        floating:
          "bg-surface/90 border-transparent text-foreground-body backdrop-blur hover:bg-surface hover:border-transparent hover:text-foreground-strong disabled:text-foreground-disabled",
        link: cn(
          linkBaseClasses,
          linkVariantClasses.secondary,
          linkFocusClasses,
          "inline-flex items-center gap-2 border-none bg-transparent px-0 py-0 text-left disabled:opacity-50",
        ),
      },
      size: {
        small: "px-3 py-2 text-body leading-5 gap-1.5 h-9 min-w-[5.5rem]",
        large: "px-4 py-2.5 text-body leading-5 gap-2 h-[2.875rem]",
        icon: "p-3 !min-w-0",
        xs: "h-7 min-w-0 gap-1.5 px-2.5 text-caption leading-4",
        "icon-xs": "size-7 min-w-0 p-0",
        "icon-sm": "size-8 min-w-0 p-0",
        "icon-2xs": "size-6 min-w-0 p-0",
        // 32px, for actions inside a list row or card.
        row: "h-8 min-w-0 gap-1.5 px-3 py-2 text-body leading-5",
        // No box: a text button inside copy, usually with variant="link".
        inline: "h-auto min-w-0 gap-1.5 p-0 text-body leading-5",
        "inline-xs": "h-auto min-w-0 gap-1 p-0 text-caption leading-4",
      },
      tone: BUTTON_TONES,
      shape: BUTTON_SHAPES,
    },
    compoundVariants: [
      {
        variant: "outline",
        size: ["xs", "icon-xs", "icon-sm"],
        class: "border-border-strong hover:border-border-strong",
      },
      {
        variant: "icon",
        size: ["icon-xs", "icon-sm"],
        class: "text-foreground-secondary",
      },
    ],
    defaultVariants: {
      variant: "primary",
      size: "large",
      shape: "pill",
    },
  },
);

export const BUTTON_ICON_SIZE = {
  small: 16,
  large: 18,
  icon: 16,
  xs: 14,
  "icon-xs": 14,
  "icon-sm": 16,
  "icon-2xs": 14,
  row: 16,
  inline: 16,
  "inline-xs": 14,
} as const;

export const ICON_ONLY_SIZES: ReadonlySet<string> = new Set([
  "icon-xs",
  "icon-sm",
  "icon-2xs",
]);

type BaseButtonProps = {
  loading?: boolean;
  /** Hugeicon rendered before the label at the size's icon scale. */
  leadingIcon?: IconSvgElement;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  asChild?: boolean;
  withTooltip?: boolean;
  /** `body-md` sets the label at the text-body-md role (15px/1.5rem) for
   *  sizes whose label is text-body, e.g. inside a profile tab. */
  textSize?: "body" | "body-md";
} & VariantProps<typeof extendedButtonVariants>;

type ButtonAsButton = BaseButtonProps &
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    as?: "button";
    href?: never;
  };

type ButtonAsLink = BaseButtonProps &
  Omit<React.AnchorHTMLAttributes<HTMLAnchorElement>, keyof LinkProps> &
  LinkProps & {
    as: "NextLink";
    disabled?: boolean;
  };

export type ButtonProps = ButtonAsButton | ButtonAsLink;
