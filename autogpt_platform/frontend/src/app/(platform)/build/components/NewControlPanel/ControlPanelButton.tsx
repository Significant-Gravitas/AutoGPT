import { cn } from "@/lib/utils";
import React from "react";

interface Props extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
  // Icon-only control: the visible tooltip is not an accessible name.
  "aria-label": string;
}

export const ControlPanelButton = React.forwardRef<HTMLButtonElement, Props>(
  function ControlPanelButton(
    { selected = false, disabled, type = "button", className, ...rest },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={type}
        disabled={disabled}
        className={cn(
          "flex w-auto items-center justify-center bg-surface px-4 py-4 whitespace-normal text-foreground-emphasis shadow-none hover:cursor-pointer hover:bg-surface-hover hover:text-foreground-base focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none focus-visible:ring-inset",
          selected &&
            "bg-accent-surface text-accent-strong hover:cursor-default hover:bg-accent-surface hover:text-accent-strong active:bg-accent-surface active:text-accent-strong",
          disabled && "cursor-not-allowed opacity-50 hover:cursor-not-allowed",
          className,
        )}
        {...rest}
      />
    );
  },
);
