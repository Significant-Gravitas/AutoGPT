import {
  linkBaseClasses,
  linkFocusClasses,
  linkVariantClasses,
} from "@/components/atoms/Link/Link";
import { cn } from "@/lib/utils";
import { cva, VariantProps } from "class-variance-authority";
import { IconSvgElement } from "@hugeicons/react";
import { LinkProps } from "next/link";

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
        ghost:
          "bg-transparent border-transparent text-foreground-base hover:bg-surface-muted hover:border-surface-muted rounded-full disabled:text-surface-active disabled:opacity-1",
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
} as const;

export const ICON_ONLY_SIZES: ReadonlySet<string> = new Set([
  "icon-xs",
  "icon-sm",
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
