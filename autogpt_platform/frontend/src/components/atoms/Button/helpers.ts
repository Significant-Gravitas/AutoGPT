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
// TODO(tokens): no role or scale value is identical for the outline border
// #a6a6a6, min-w-[7.7rem], min-w-[5.5rem], h-[2.875rem] or rounded-[96px]
// (rounded-full is 9999px), so they stay raw.
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
          "bg-transparent border-[#a6a6a6] text-foreground-base hover:bg-surface-hover hover:border-[#a6a6a6] rounded-full disabled:border-border-base disabled:text-surface-active disabled:opacity-1",
        ghost:
          "bg-transparent border-transparent text-foreground-base hover:bg-surface-muted hover:border-surface-muted rounded-full disabled:text-surface-active disabled:opacity-1",
        icon: "bg-surface text-foreground-emphasis border border-border-base shadow-sm hover:bg-surface-muted hover:border-border-strong rounded-[96px] disabled:shadow-none disabled:opacity-1 !min-w-0",
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
        class: "border-[#a6a6a6] hover:border-[#a6a6a6]",
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

// The rounded-md shadcn looks the app drew before the design system, kept at
// today's pixels until each screen moves to one of the variants above. Picking a
// `legacy-*` variant renders a plain <button> with these classes and the
// children as they are.
// TODO(tokens): no role is neutral-900 #171717, neutral-200 #E5E5E5 or
// neutral-100 #F5F5F5. legacy-default's text-neutral-50 #FAFAFA stays raw:
// `primary-foreground` holds that value in :root but is 24 24 27 under .dark,
// so it would not render the same there (`background`, `sidebar` and
// `sidebar-primary-foreground` hold it too, for other roles). The dark: classes
// never apply (forcedTheme="light") and are DS6's (SECRT-3116) to delete.
export const legacyButtonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-body leading-5 font-medium transition-colors focus-visible:outline-none focus-visible:ring-0 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        "legacy-default":
          "bg-neutral-900 text-neutral-50 shadow hover:bg-neutral-900/90 dark:bg-neutral-50 dark:text-neutral-900 dark:hover:bg-neutral-50/90",
        "legacy-destructive":
          "bg-error text-destructive-foreground shadow-sm hover:bg-error/90 dark:bg-red-900 dark:text-neutral-50 dark:hover:bg-red-900/90",
        "legacy-outline":
          "border border-neutral-200 bg-surface shadow-sm hover:bg-neutral-100 hover:text-neutral-900 dark:border-neutral-800 dark:bg-neutral-950 dark:hover:bg-neutral-800 dark:hover:text-neutral-50",
        "legacy-ghost":
          "hover:bg-neutral-100 hover:text-neutral-900 dark:hover:bg-neutral-800 dark:hover:text-neutral-50",
        "legacy-link":
          "text-neutral-900 underline-offset-4 hover:underline dark:text-neutral-50",
      },
      size: {
        "legacy-default": "h-9 px-4 py-2",
        "legacy-sm": "h-8 rounded-md px-3 text-caption leading-4",
        "legacy-icon": "h-9 w-9",
      },
    },
    defaultVariants: {
      size: "legacy-default",
    },
  },
);

type LegacyButtonVariant = NonNullable<
  VariantProps<typeof legacyButtonVariants>["variant"]
>;

const LEGACY_BUTTON_VARIANTS: ReadonlySet<string> =
  new Set<LegacyButtonVariant>([
    "legacy-default",
    "legacy-destructive",
    "legacy-outline",
    "legacy-ghost",
    "legacy-link",
  ]);

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

export type LegacyButtonProps = Omit<
  React.ButtonHTMLAttributes<HTMLButtonElement>,
  "size"
> & {
  variant: LegacyButtonVariant;
  size?: VariantProps<typeof legacyButtonVariants>["size"];
  ref?: React.Ref<HTMLButtonElement>;
  as?: "button";
  href?: never;
};

export type ButtonProps = ButtonAsButton | ButtonAsLink | LegacyButtonProps;

export function isLegacyButton(props: ButtonProps): props is LegacyButtonProps {
  return props.variant != null && LEGACY_BUTTON_VARIANTS.has(props.variant);
}
