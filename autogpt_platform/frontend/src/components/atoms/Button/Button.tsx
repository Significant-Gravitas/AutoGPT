"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Button as KobraButton } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import NextLink, { type LinkProps } from "next/link";
import React, { forwardRef } from "react";
import {
  BUTTON_ICON_SIZE,
  ButtonProps,
  ICON_ONLY_SIZES,
  resolveButtonStyle,
} from "./helpers";

export const Button = forwardRef<
  HTMLButtonElement | HTMLAnchorElement,
  ButtonProps
>(function Button(props, ref) {
  const {
    className,
    variant,
    size,
    loading = false,
    withTooltip = true,
    leadingIcon,
    leftIcon,
    rightIcon,
    children,
    as = "button",
    unmask = true,
    asChild: _asChild, // Destructure to prevent passing to DOM
    ...restProps
  } = props;

  const isDisabled = "disabled" in props ? Boolean(props.disabled) : false;
  const ariaLabel =
    "aria-label" in restProps ? restProps["aria-label"] : undefined;
  const isIconOnly =
    variant === "icon" || (size != null && ICON_ONLY_SIZES.has(size));
  const shouldShowTooltip = isIconOnly && ariaLabel && !loading && withTooltip;

  const style = resolveButtonStyle({ variant, size });
  const resolvedLeftIcon = leadingIcon ? (
    <Icon
      icon={leadingIcon}
      size={BUTTON_ICON_SIZE[size ?? "lg"]}
      aria-hidden
    />
  ) : (
    leftIcon
  );

  const content = (
    <>
      {loading ? <Spinner aria-hidden /> : resolvedLeftIcon}
      {children}
      {!loading && rightIcon}
    </>
  );

  const shared = {
    variant: style.variant,
    size: style.size,
    rounded: style.rounded,
    flat: style.flat,
    disabled: isDisabled || loading,
    "aria-busy": loading || undefined,
  };

  const element =
    as === "NextLink" ? (
      <KobraButton
        ref={ref as React.Ref<HTMLButtonElement>}
        {...shared}
        className={cn(
          style.className,
          className,
          (isDisabled || loading) && "pointer-events-none opacity-50",
          unmask && "sentry-unmask",
        )}
        nativeButton={false}
        role={undefined}
        render={<NextLink {...(restProps as LinkProps)} />}
      >
        {content}
      </KobraButton>
    ) : (
      <KobraButton
        ref={ref as React.Ref<HTMLButtonElement>}
        {...shared}
        className={cn(style.className, className, unmask && "sentry-unmask")}
        {...(restProps as React.ButtonHTMLAttributes<HTMLButtonElement>)}
      >
        {content}
      </KobraButton>
    );

  if (!shouldShowTooltip) return element;

  return (
    <Tooltip>
      <TooltipTrigger render={element} />
      <TooltipContent>{ariaLabel}</TooltipContent>
    </Tooltip>
  );
});
