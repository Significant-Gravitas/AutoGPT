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

// A ghost Button's box: no fill or border until hover. The ghost-* variants
// add a quieter text colour to it.
const GHOST =
  "bg-transparent border-transparent hover:bg-surface-muted hover:border-surface-muted rounded-full disabled:text-surface-active disabled:opacity-1";

// Extended button variants based on our design system.
// TODO(tokens): no spacing step is min-w-[7.7rem], min-w-[5.5rem] or
// h-[2.875rem], so they stay raw.
export const extendedButtonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus disabled:pointer-events-none disabled:opacity-50 font-sans leading-snug border min-w-[7.7rem]",
  {
    variants: {
      variant: {
        primary:
          "bg-action border-action text-action-foreground hover:bg-action-hover hover:border-action-hover rounded-full disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base disabled:opacity-1",
        secondary:
          "bg-surface border-border-base text-foreground-emphasis shadow-sm hover:bg-surface-muted hover:border-border-strong rounded-full disabled:text-foreground-disabled disabled:bg-surface-muted disabled:border-separator disabled:shadow-none disabled:opacity-1",
        destructive:
          "bg-error border-error text-action-foreground hover:bg-error-hover hover:border-error-hover rounded-full disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base disabled:opacity-1",
        outline:
          "bg-transparent border-border-strong text-foreground-base hover:bg-surface-hover hover:border-border-strong rounded-full disabled:border-border-base disabled:text-surface-active disabled:opacity-1",
        ghost: `${GHOST} text-foreground-base`,
        // Secondary actions in a row or card: grey until hovered.
        "ghost-muted": `${GHOST} text-foreground-tertiary hover:text-foreground-body`,
        "ghost-subtle": `${GHOST} text-foreground-subtle hover:text-foreground-body`,
        // A remove or delete action that turns red on hover.
        "ghost-danger": `${GHOST} text-foreground-tertiary hover:text-error`,
        "ghost-warning": `${GHOST} text-warning-foreground hover:bg-warning-surface hover:border-warning-surface`,
        // An upgrade or promotion call to action.
        brand:
          "bg-brand border-brand text-action-foreground hover:bg-accent-strong hover:border-accent-strong rounded-full disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base disabled:opacity-100",
        // Goes ahead despite a warning, e.g. an update that breaks links.
        warning:
          "bg-warning-foreground border-warning-foreground text-action-foreground hover:bg-warning-foreground/90 hover:border-warning-foreground rounded-full disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base disabled:opacity-100",
        // A secondary Button showing that its action worked, e.g. "Copied".
        success:
          "bg-success/30 border-success text-foreground-base shadow-sm hover:bg-success/40 hover:border-success rounded-full disabled:shadow-none disabled:opacity-100",
        // An icon action in a toolbar such as the chat composer: no box
        // until hovered, faded while disabled.
        tray: "bg-transparent border-transparent text-foreground-base hover:bg-surface-hover hover:border-transparent rounded-full disabled:opacity-40 !min-w-0",
        icon: "bg-surface text-foreground-emphasis border border-border-base shadow-sm hover:bg-surface-muted hover:border-border-strong rounded-full disabled:shadow-none disabled:opacity-1 !min-w-0",
        toggle:
          "bg-transparent border-transparent text-foreground-tertiary hover:bg-transparent hover:border-transparent hover:text-foreground-emphasis aria-pressed:bg-surface-hover aria-pressed:text-foreground-strong rounded-md disabled:opacity-50",
        floating:
          "bg-surface/90 border-transparent text-foreground-body backdrop-blur hover:bg-surface hover:border-transparent hover:text-foreground-strong rounded-md disabled:text-foreground-disabled disabled:opacity-1",
        link: cn(
          linkBaseClasses,
          linkVariantClasses.secondary,
          linkFocusClasses,
          "inline-flex items-center gap-2 border-none bg-transparent px-0 py-0 text-left",
        ),
      },
      size: {
        small: "px-3 py-2 text-body leading-5 gap-1.5 h-9 min-w-[5.5rem]",
        large: "px-4 py-2.5 text-body leading-5 gap-2 h-[2.875rem]",
        icon: "p-3 !min-w-0",
        xs: "h-7 min-w-0 gap-1.5 rounded-md px-2.5 text-caption leading-4",
        "icon-xs": "size-7 min-w-0 rounded-md p-0",
        "icon-sm": "size-8 min-w-0 rounded-lg p-0",
        "icon-2xs": "size-6 min-w-0 rounded-md p-0",
        // 32px, for actions inside a list row or card.
        row: "h-8 min-w-0 gap-1.5 px-3 py-2 text-body leading-5",
        // No box: a text button inside copy, usually with variant="link".
        inline: "h-auto min-w-0 gap-1.5 p-0 text-body leading-5",
        "inline-xs": "h-auto min-w-0 gap-1 p-0 text-caption leading-4",
      },
      // Text colour over the variant's own (see src/components/styles/tones.ts).
      tone: tones,
      // Corner radius over the variant's and size's own.
      shape: {
        pill: "rounded-full",
        rounded: "rounded-xl",
        soft: "rounded-lg",
      },
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
