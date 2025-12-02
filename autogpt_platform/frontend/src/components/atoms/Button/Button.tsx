import { cn } from "@/lib/utils";
import { CircleNotchIcon } from "@phosphor-icons/react/dist/ssr";
import NextLink, { type LinkProps } from "next/link";
import React from "react";
import { ButtonProps, extendedButtonVariants } from "./helpers";

export function Button(props: ButtonProps) {
  const {
    className,
    variant,
    size,
    loading = false,
    leftIcon,
    rightIcon,
    children,
    as = "button",
    ...restProps
  } = props;

  const disabled = "disabled" in props ? props.disabled : false;
  const isDisabled = disabled;

  const buttonContent = (
    <>
      {loading && (
        <CircleNotchIcon className="h-4 w-4 animate-spin" weight="bold" />
      )}
      {!loading && leftIcon}
      {children}
      {!loading && rightIcon}
    </>
  );

  if (variant === "link") {
    const buttonRest = { ...(restProps as Record<string, unknown>) };

    if ("href" in buttonRest) {
      delete buttonRest.href;
    }

    return (
      <button
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
            "pointer-events-none border-zinc-500 bg-zinc-500 text-white",
          );

    return as === "NextLink" ? (
      <NextLink
        {...(restProps as LinkProps)}
        className={loadingClassName}
        aria-disabled="true"
      >
        <CircleNotchIcon className="h-4 w-4 animate-spin" weight="bold" />
        {children}
      </NextLink>
    ) : (
      <button className={loadingClassName} disabled>
        <CircleNotchIcon className="h-4 w-4 animate-spin" weight="bold" />
        {children}
      </button>
    );
  }

  if (as === "NextLink") {
    return (
      <NextLink
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
  }

  return (
    <button
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
}
