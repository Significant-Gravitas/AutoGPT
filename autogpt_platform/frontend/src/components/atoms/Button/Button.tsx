import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import NextLink, { type LinkProps } from "next/link";
import React from "react";
import {
  BUTTON_ICON_SIZE,
  ButtonProps,
  extendedButtonVariants,
  ICON_ONLY_SIZES,
  isLegacyButton,
} from "./helpers";
import { LegacyButton } from "./LegacyButton";
import { CompactButtonProps, isCompactButton } from "./compact-variants";
import { CompactButton } from "./CompactButton";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function Button(props: ButtonProps | CompactButtonProps) {
  if (isCompactButton(props)) return <CompactButton {...props} />;
  if (isLegacyButton(props)) return <LegacyButton {...props} />;

  const {
    className: callerClassName,
    textSize = "body",
    variant,
    size,
    loading = false,
    withTooltip = true,
    leadingIcon,
    leftIcon,
    rightIcon,
    children,
    as = "button",
    asChild: _asChild, // Destructure to prevent passing to DOM
    ...restProps
  } = props;
  const className =
    textSize === "body-md"
      ? cn(callerClassName, "text-body-md")
      : callerClassName;

  const disabled = "disabled" in props ? props.disabled : false;
  const isDisabled = disabled;

  // Extract aria-label for tooltip on icon variant
  const ariaLabel =
    "aria-label" in restProps ? restProps["aria-label"] : undefined;

  const isIconOnly =
    variant === "icon" || (size != null && ICON_ONLY_SIZES.has(size));
  const shouldShowTooltip = isIconOnly && ariaLabel && !loading && withTooltip;
  const resolvedLeftIcon = leadingIcon ? (
    <Icon
      data-slot="button-icon"
      icon={leadingIcon}
      size={BUTTON_ICON_SIZE[size ?? "large"]}
      aria-hidden
    />
  ) : (
    leftIcon
  );

  // Helper to wrap button with tooltip if needed
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
          className="h-4 w-4 animate-spin"
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
          extendedButtonVariants({ variant: "link", className }),
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
    const loadingClassName =
      variant === "ghost"
        ? cn(
            extendedButtonVariants({ variant, size, className }),
            "pointer-events-none",
          )
        : cn(
            extendedButtonVariants({ variant: "primary", size, className }),
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

    // Spread first so `className` and `disabled` below still win. Without this
    // the loading branch silently drops every extra prop the caller passed —
    // aria-label, data-testid, analytics data-* — the moment a click flips it
    // into loading, which is exactly when a click listener needs to read them.
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
          extendedButtonVariants({ variant, size, className }),
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
        extendedButtonVariants({ variant, size, className }),
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
