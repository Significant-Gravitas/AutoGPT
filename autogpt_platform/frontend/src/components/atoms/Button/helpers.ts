import {
  linkBaseClasses,
  linkFocusClasses,
  linkVariantClasses,
} from "@/components/atoms/Link/Link";
import { cn } from "@/lib/utils";
import { cva, VariantProps } from "class-variance-authority";
import { IconSvgElement } from "@hugeicons/react";
import { LinkProps } from "next/link";

// Extended button variants based on our design system
export const extendedButtonVariants = cva(
  "inline-flex items-center justify-center border font-sans leading-snug font-medium whitespace-nowrap focus-ring transition-colors focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        primary:
          "rounded-full border-primary bg-primary text-primary-foreground hover:border-primary/90 hover:bg-primary/90",
        secondary:
          "rounded-full border-border bg-background text-secondary-foreground shadow-xs hover:bg-muted",
        destructive:
          "rounded-full border-destructive bg-destructive text-destructive-foreground hover:border-destructive/90 hover:bg-destructive/90",
        outline:
          "rounded-full border-border bg-transparent text-foreground hover:bg-muted",
        ghost:
          "rounded-full border-transparent bg-transparent text-foreground hover:bg-muted",
        icon: "min-w-0! rounded-full border-border bg-card text-foreground shadow-xs hover:bg-muted",
        toggle:
          "rounded-md border-transparent bg-transparent text-muted-foreground hover:text-foreground aria-pressed:bg-muted aria-pressed:text-foreground",
        floating:
          "rounded-md border-transparent bg-card/90 text-muted-foreground backdrop-blur-sm hover:bg-card hover:text-foreground",
        link: cn(
          linkBaseClasses,
          linkVariantClasses.secondary,
          linkFocusClasses,
          "inline-flex items-center gap-2 border-none bg-transparent px-0 py-0 text-left",
        ),
      },
      size: {
        sm: "h-8 gap-1.5 rounded-md px-3 text-xs",
        md: "h-9 min-w-22 gap-1.5 px-3 text-sm",
        lg: "h-10 min-w-30 gap-2 px-4 text-sm",
        "icon-sm": "size-8 rounded-md p-0",
        "icon-md": "size-9 p-0",
        "icon-lg": "size-10 p-0",
      },
    },
    compoundVariants: [
      {
        variant: "icon",
        size: ["icon-sm", "icon-md"],
        class: "text-muted-foreground",
      },
    ],
    defaultVariants: {
      variant: "primary",
      size: "lg",
    },
  },
);

export const BUTTON_ICON_SIZE = {
  sm: 14,
  md: 16,
  lg: 18,
  "icon-sm": 16,
  "icon-md": 16,
  "icon-lg": 18,
} as const;

// Sizes that get the aria-label tooltip automatically (as variant="icon"
// does). icon-md and icon-lg replace the old `icon` size, which did not.
export const ICON_ONLY_SIZES: ReadonlySet<string> = new Set(["icon-sm"]);

type BaseButtonProps = {
  loading?: boolean;
  /** Hugeicon rendered before the label at the size's icon scale. */
  leadingIcon?: IconSvgElement;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  asChild?: boolean;
  withTooltip?: boolean;
  /**
   * Adds the sentry-unmask class for static button labels.
   * Disable for user-provided or dynamic strings.
   */
  unmask?: boolean;
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
