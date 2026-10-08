import {
  linkBaseClasses,
  linkFocusClasses,
  linkVariantClasses,
} from "@/components/atoms/Link/Link";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { IconSvgElement } from "@hugeicons/react";
import { LinkProps } from "next/link";
import type { VariantProps } from "class-variance-authority";

type KobraButtonVariants = VariantProps<typeof buttonVariants>;
type KobraVariant = NonNullable<KobraButtonVariants["variant"]>;
type KobraSize = NonNullable<KobraButtonVariants["size"]>;

export type ButtonVariant =
  | "primary"
  | "secondary"
  | "destructive"
  | "outline"
  | "ghost"
  | "icon"
  | "toggle"
  | "floating"
  | "link";

export type ButtonSize = "sm" | "md" | "lg" | "icon-sm" | "icon-md" | "icon-lg";

interface ButtonVariantOptions {
  variant?: ButtonVariant | null;
  size?: ButtonSize | null;
  className?: string;
}

// House variant → Kobra variant, plus the house styling Kobra has no
// variant for (toggle's aria-pressed state, floating's glass card, the
// Link atom's typography for `link`).
const VARIANT_MAP: Record<
  ButtonVariant,
  {
    variant: KobraVariant;
    rounded: boolean;
    flat?: boolean;
    className?: string;
  }
> = {
  primary: { variant: "default", rounded: true },
  secondary: { variant: "outline", rounded: true },
  destructive: { variant: "destructive", rounded: true },
  outline: { variant: "outline", rounded: true, flat: true },
  ghost: { variant: "ghost", rounded: true },
  icon: { variant: "outline", rounded: true, className: "min-w-0!" },
  toggle: {
    variant: "ghost",
    rounded: false,
    className:
      "rounded-md text-muted-foreground hover:text-foreground aria-pressed:bg-muted aria-pressed:text-foreground",
  },
  floating: {
    variant: "ghost",
    rounded: false,
    className:
      "rounded-md bg-card/90 text-muted-foreground backdrop-blur-sm hover:bg-card hover:text-foreground",
  },
  link: {
    variant: "link",
    rounded: false,
    className: cn(
      linkBaseClasses,
      linkVariantClasses.secondary,
      linkFocusClasses,
      "inline-flex items-center gap-2 border-none bg-transparent px-0 py-0 text-left",
    ),
  },
};

// House size → Kobra size. Kobra's controls are 28/32/36px; the house keeps
// 32/36/40 (DESIGN.md), its min-widths, and 14/16/18px icons, so those are
// forced through className. `sm` and `icon-sm` are rounded-rectangle chips.
const SIZE_MAP: Record<ButtonSize, { size: KobraSize; className: string }> = {
  sm: { size: "sm", className: "h-8 gap-1.5 rounded-md px-3 text-xs" },
  md: { size: "default", className: "h-9 min-w-22 gap-1.5 px-3 text-sm" },
  lg: {
    size: "lg",
    className:
      "h-10 min-w-30 gap-2 px-4 text-sm [&_svg:not([class*='size-'])]:size-4.5",
  },
  "icon-sm": { size: "icon-sm", className: "size-8 rounded-md p-0" },
  "icon-md": { size: "icon", className: "size-9 p-0" },
  "icon-lg": {
    size: "icon-lg",
    className: "size-10 p-0 [&_svg:not([class*='size-'])]:size-4.5",
  },
};

const CHIP_SIZES: ReadonlySet<ButtonSize> = new Set(["sm", "icon-sm"]);

export function resolveButtonStyle({
  variant,
  size,
}: Omit<ButtonVariantOptions, "className">) {
  const houseVariant = variant ?? "primary";
  const houseSize = size ?? "lg";
  const mappedVariant = VARIANT_MAP[houseVariant];
  const mappedSize = SIZE_MAP[houseSize];
  return {
    variant: mappedVariant.variant,
    size: mappedSize.size,
    rounded: mappedVariant.rounded && !CHIP_SIZES.has(houseSize),
    flat: mappedVariant.flat ?? false,
    className: cn(
      mappedSize.className,
      mappedVariant.className,
      houseVariant === "icon" &&
        (houseSize === "icon-sm" || houseSize === "icon-md") &&
        "text-muted-foreground",
    ),
  };
}

/** The house button classes as a string, for elements that cannot be a Button. */
export function extendedButtonVariants({
  variant,
  size,
  className,
}: ButtonVariantOptions = {}) {
  const style = resolveButtonStyle({ variant, size });
  return cn(
    buttonVariants({
      variant: style.variant,
      size: style.size,
      rounded: style.rounded,
      flat: style.flat,
    }),
    style.className,
    className,
  );
}

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
