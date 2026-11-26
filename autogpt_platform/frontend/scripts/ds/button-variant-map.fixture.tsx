/**
 * The Button atom before its variants were reduced: helpers.ts,
 * compact-variants.ts, CompactButton.tsx and Button.tsx as they were at
 * 7fda55e2bc, in one file, with the Link atom's classes written out.
 * button-variant-map.test.tsx renders this OldButton next to the Button atom
 * for every combination in use. Never edit it.
 */
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Icon } from "@/components/atoms/Icon/Icon";
import { tones } from "@/components/styles/tones";
import { cn } from "@/lib/utils";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { cva, type VariantProps } from "class-variance-authority";
import NextLink, { type LinkProps } from "next/link";
import type React from "react";

const linkBaseClasses = "font-sans text-body font-medium text-foreground-base";
const linkSecondaryClasses = "underline";
const linkFocusClasses =
  "focus:outline-none focus:ring-2 focus:ring-info focus:ring-offset-2 rounded-sm";

const GHOST =
  "bg-transparent border-transparent hover:bg-surface-muted hover:border-surface-muted rounded-pill disabled:text-surface-active disabled:opacity-1";

const OLD_VARIANTS = {
  variant: {
    primary:
      "bg-action border-action text-action-foreground hover:bg-action-hover hover:border-action-hover rounded-pill disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base disabled:opacity-1",
    secondary:
      "bg-surface border-border-base text-foreground-emphasis shadow-sm hover:bg-surface-muted hover:border-border-strong rounded-pill disabled:text-foreground-disabled disabled:bg-surface-muted disabled:border-separator disabled:shadow-none disabled:opacity-1",
    destructive:
      "bg-error border-error text-action-foreground hover:bg-error-hover hover:border-error-hover rounded-pill disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base disabled:opacity-1",
    outline:
      "bg-transparent border-border-strong text-foreground-base hover:bg-surface-hover hover:border-border-strong rounded-pill disabled:border-border-base disabled:text-surface-active disabled:opacity-1",
    ghost: `${GHOST} text-foreground-base`,
    "ghost-muted": `${GHOST} text-foreground-tertiary hover:text-foreground-body`,
    "ghost-subtle": `${GHOST} text-foreground-subtle hover:text-foreground-body`,
    "ghost-danger": `${GHOST} text-foreground-tertiary hover:text-error`,
    "ghost-warning": `${GHOST} text-warning-foreground hover:bg-warning-surface hover:border-warning-surface`,
    brand:
      "bg-brand border-brand text-action-foreground hover:bg-accent-strong hover:border-accent-strong rounded-pill disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base disabled:opacity-100",
    warning:
      "bg-warning-foreground border-warning-foreground text-action-foreground hover:bg-warning-foreground/90 hover:border-warning-foreground rounded-pill disabled:text-action-foreground disabled:bg-surface-active disabled:border-border-base disabled:opacity-100",
    success:
      "bg-success/30 border-success text-foreground-base shadow-sm hover:bg-success/40 hover:border-success rounded-pill disabled:shadow-none disabled:opacity-100",
    tray: "bg-transparent border-transparent text-foreground-base hover:bg-surface-hover hover:border-transparent rounded-pill disabled:opacity-40 !min-w-0",
    icon: "bg-surface text-foreground-emphasis border border-border-base shadow-sm hover:bg-surface-muted hover:border-border-strong rounded-pill disabled:shadow-none disabled:opacity-1 !min-w-0",
    toggle:
      "bg-transparent border-transparent text-foreground-tertiary hover:bg-transparent hover:border-transparent hover:text-foreground-emphasis aria-pressed:bg-surface-hover aria-pressed:text-foreground-strong rounded-md disabled:opacity-50",
    floating:
      "bg-surface/90 border-transparent text-foreground-body backdrop-blur hover:bg-surface hover:border-transparent hover:text-foreground-strong rounded-md disabled:text-foreground-disabled disabled:opacity-1",
    link: cn(
      linkBaseClasses,
      linkSecondaryClasses,
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
    row: "h-8 min-w-0 gap-1.5 px-3 py-2 text-body leading-5",
    inline: "h-auto min-w-0 gap-1.5 p-0 text-body leading-5",
    "inline-xs": "h-auto min-w-0 gap-1 p-0 text-caption leading-4",
  },
  tone: tones,
  shape: {
    pill: "rounded-pill",
    rounded: "rounded-xl",
    soft: "rounded-lg",
  },
};

export const OLD_VARIANT_NAMES = Object.keys(OLD_VARIANTS.variant);
export const OLD_SIZE_NAMES = Object.keys(OLD_VARIANTS.size);
export const OLD_SHAPE_NAMES = Object.keys(OLD_VARIANTS.shape);

export const oldButtonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus disabled:pointer-events-none disabled:opacity-50 font-sans leading-snug border min-w-[7.7rem]",
  {
    variants: OLD_VARIANTS,
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

const OLD_BUTTON_ICON_SIZE = {
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

const OLD_ICON_ONLY_SIZES: ReadonlySet<string> = new Set([
  "icon-xs",
  "icon-sm",
  "icon-2xs",
]);

const COMPACT_BASE =
  "gap-2 rounded-md border-0 min-w-0 text-body leading-5 focus-visible:ring-focus [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0";

const oldCompactButtonVariants = cva(
  oldButtonVariants({ variant: null, size: null }),
  {
    variants: {
      variant: {
        "compact-primary": `${COMPACT_BASE} bg-action text-action-foreground shadow hover:bg-action-hover`,
        "compact-destructive": `${COMPACT_BASE} bg-error text-action-foreground shadow-sm hover:bg-error/90`,
        "compact-outline": `${COMPACT_BASE} border border-border-base bg-surface shadow-sm hover:bg-surface-hover hover:text-foreground-strong`,
        "compact-ghost": `${COMPACT_BASE} hover:bg-surface-hover hover:text-foreground-strong aria-pressed:bg-surface-hover`,
      },
      size: {
        compact: "h-9 px-4 py-2",
        "compact-sm": "h-8 px-3 text-caption leading-4",
        "compact-icon": "h-9 w-9",
        "compact-icon-sm": "h-8 w-8",
      },
    },
  },
);

type OldCompactVariant = NonNullable<
  VariantProps<typeof oldCompactButtonVariants>["variant"]
>;

const OLD_COMPACT_VARIANTS: ReadonlySet<string> = new Set<OldCompactVariant>([
  "compact-primary",
  "compact-destructive",
  "compact-outline",
  "compact-ghost",
]);

type OldCompactProps = Omit<
  React.ButtonHTMLAttributes<HTMLButtonElement>,
  "size"
> & {
  variant: OldCompactVariant;
  size: NonNullable<VariantProps<typeof oldCompactButtonVariants>["size"]>;
  as?: "button";
  href?: never;
};

type OldBaseProps = {
  loading?: boolean;
  leadingIcon?: IconSvgElement;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  asChild?: boolean;
  withTooltip?: boolean;
  textSize?: "body" | "body-md";
} & VariantProps<typeof oldButtonVariants>;

type OldAsButton = OldBaseProps &
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    as?: "button";
    href?: never;
  };

type OldAsLink = OldBaseProps &
  Omit<React.AnchorHTMLAttributes<HTMLAnchorElement>, keyof LinkProps> &
  LinkProps & {
    as: "NextLink";
    disabled?: boolean;
  };

export type OldButtonProps = OldAsButton | OldAsLink | OldCompactProps;

function isOldCompact(props: { variant?: unknown }): props is OldCompactProps {
  return (
    typeof props.variant === "string" && OLD_COMPACT_VARIANTS.has(props.variant)
  );
}

function OldCompactButton({
  className,
  variant,
  size,
  disabled,
  as: _as,
  ...props
}: OldCompactProps) {
  return (
    <button
      data-slot="button"
      className={cn(oldCompactButtonVariants({ variant, size, className }))}
      disabled={disabled}
      {...props}
    />
  );
}

export function OldButton(props: OldButtonProps) {
  if (isOldCompact(props)) return <OldCompactButton {...props} />;

  const {
    className: callerClassName,
    textSize = "body",
    variant,
    size,
    shape,
    tone,
    loading = false,
    withTooltip = true,
    leadingIcon,
    leftIcon,
    rightIcon,
    children,
    as = "button",
    asChild: _asChild,
    ...restProps
  } = props;
  const className =
    textSize === "body-md"
      ? cn(callerClassName, "text-body-md")
      : callerClassName;

  const disabled = "disabled" in props ? props.disabled : false;
  const isDisabled = disabled;

  const ariaLabel =
    "aria-label" in restProps ? restProps["aria-label"] : undefined;

  const isIconOnly =
    variant === "icon" || (size != null && OLD_ICON_ONLY_SIZES.has(size));
  const shouldShowTooltip = isIconOnly && ariaLabel && !loading && withTooltip;
  const resolvedLeftIcon = leadingIcon ? (
    <Icon
      data-slot="button-icon"
      icon={leadingIcon}
      size={OLD_BUTTON_ICON_SIZE[size ?? "large"]}
      aria-hidden
    />
  ) : (
    leftIcon
  );

  const wrapWithTooltip = (buttonElement: React.ReactElement) => {
    if (shouldShowTooltip) {
      return (
        <Tooltip>
          <TooltipTrigger asChild>{buttonElement}</TooltipTrigger>
          <TooltipContent data-slot="button-tooltip">
            {ariaLabel}
          </TooltipContent>
        </Tooltip>
      );
    }
    return buttonElement;
  };

  const buttonContent = (
    <>
      {loading && (
        <Icon
          data-slot="button-spinner"
          icon={Loading03Icon}
          spin
          className="h-4 w-4"
        />
      )}
      {!loading && resolvedLeftIcon}
      <span data-slot="button-label" className="contents">
        {children}
      </span>
      {!loading && rightIcon}
    </>
  );

  if (variant === "link") {
    const buttonRest = { ...(restProps as Record<string, unknown>) };

    if ("href" in buttonRest) {
      delete buttonRest.href;
    }

    const linkButton = (
      <button
        data-slot="button"
        className={cn(
          oldButtonVariants({
            variant: "link",
            size,
            shape,
            tone,
            className,
          }),
          loading && "pointer-events-none opacity-60",
          isDisabled && "pointer-events-none opacity-50",
        )}
        disabled={isDisabled || loading}
        {...(buttonRest as React.ButtonHTMLAttributes<HTMLButtonElement>)}
      >
        {buttonContent}
      </button>
    );

    return wrapWithTooltip(linkButton);
  }

  if (loading) {
    const loadingClassName = variant?.startsWith("ghost")
      ? cn(
          oldButtonVariants({ variant, size, shape, tone, className }),
          "pointer-events-none",
        )
      : cn(
          oldButtonVariants({
            variant: "primary",
            size,
            shape,
            className,
          }),
          "pointer-events-none border-foreground-tertiary bg-foreground-tertiary text-action-foreground",
        );

    if (as === "NextLink") {
      return (
        <NextLink
          data-slot="button"
          {...(restProps as LinkProps)}
          className={loadingClassName}
          aria-disabled="true"
        >
          {buttonContent}
        </NextLink>
      );
    }

    const loadingButton = (
      <button
        data-slot="button"
        {...(restProps as React.ButtonHTMLAttributes<HTMLButtonElement>)}
        className={loadingClassName}
        disabled
      >
        {buttonContent}
      </button>
    );

    return wrapWithTooltip(loadingButton);
  }

  if (as === "NextLink") {
    const nextLinkButton = (
      <NextLink
        data-slot="button"
        {...(restProps as LinkProps)}
        className={cn(
          oldButtonVariants({ variant, size, shape, tone, className }),
          loading && "pointer-events-none",
          isDisabled && "pointer-events-none opacity-50",
        )}
        aria-disabled={isDisabled}
      >
        {buttonContent}
      </NextLink>
    );

    return wrapWithTooltip(nextLinkButton);
  }

  const regularButton = (
    <button
      data-slot="button"
      className={cn(
        oldButtonVariants({ variant, size, shape, tone, className }),
        loading && "pointer-events-none",
      )}
      disabled={isDisabled}
      {...(restProps as React.ButtonHTMLAttributes<HTMLButtonElement>)}
    >
      {buttonContent}
    </button>
  );

  return wrapWithTooltip(regularButton);
}
