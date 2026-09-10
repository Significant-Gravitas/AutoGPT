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
  "inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neutral-950 disabled:pointer-events-none disabled:opacity-50 font-sans leading-snug border min-w-[7.7rem]",
  {
    variants: {
      variant: {
        primary:
          "bg-zinc-800 border-zinc-800 text-white hover:bg-zinc-900 hover:border-zinc-900 rounded-full disabled:text-white disabled:bg-zinc-200 disabled:border-zinc-200 disabled:opacity-1",
        secondary:
          "bg-zinc-100 border-zinc-100 text-black hover:bg-zinc-200 hover:border-zinc-200 rounded-full disabled:text-zinc-300 disabled:bg-zinc-50 disabled:border-zinc-50 disabled:opacity-1",
        destructive:
          "bg-red-500 border-red-500 text-white hover:bg-red-600 hover:border-red-600 rounded-full disabled:text-white disabled:bg-zinc-200 disabled:border-zinc-200 disabled:opacity-1",
        outline:
          "bg-transparent border-zinc-700 text-black hover:bg-zinc-100 hover:border-zinc-700 rounded-full disabled:border-zinc-200 disabled:text-zinc-200 disabled:opacity-1",
        ghost:
          "bg-transparent border-transparent text-black hover:bg-zinc-50 hover:border-zinc-50 rounded-full disabled:text-zinc-200 disabled:opacity-1",
        icon: "bg-transparent text-black border border-zinc-300 hover:bg-zinc-100 hover:border-zinc-600 rounded-[96px] disabled:opacity-1 !min-w-0",
        toggle:
          "bg-transparent border-transparent text-zinc-500 hover:bg-transparent hover:border-transparent hover:text-zinc-800 aria-pressed:bg-zinc-100 aria-pressed:text-zinc-900 rounded-md disabled:opacity-50",
        floating:
          "bg-white/90 border-transparent text-zinc-700 backdrop-blur hover:bg-white hover:border-transparent hover:text-zinc-900 rounded-md disabled:text-zinc-300 disabled:opacity-1",
        link: cn(
          linkBaseClasses,
          linkVariantClasses.secondary,
          linkFocusClasses,
          "inline-flex items-center gap-2 border-none bg-transparent px-0 py-0 text-left",
        ),
      },
      size: {
        small: "px-3 py-2 text-sm gap-1.5 h-[2.25rem] min-w-[5.5rem]",
        large: "px-4 py-3 text-sm gap-2 h-[3.25rem]",
        icon: "p-3 !min-w-0",
        xs: "h-7 min-w-0 gap-1.5 rounded-md px-2.5 text-xs",
        "icon-xs": "size-7 min-w-0 rounded-md p-0",
        "icon-sm": "size-8 min-w-0 rounded-lg p-0",
      },
    },
    compoundVariants: [
      {
        variant: "outline",
        size: ["xs", "icon-xs", "icon-sm"],
        class: "border-zinc-200 hover:border-zinc-300",
      },
      {
        variant: "icon",
        size: ["icon-xs", "icon-sm"],
        class:
          "border-zinc-200 text-zinc-600 hover:border-zinc-300 hover:bg-zinc-50",
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
