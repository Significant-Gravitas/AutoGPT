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
          // TODO(tokens): no fitting role for hover:text-zinc-950 #09090B; #09090B is foreground, whose .dark value (250 250 250) differs (QUALITY 43, SECRT-3154)
          "flex w-auto items-center justify-center whitespace-normal bg-surface px-4 py-4 text-foreground-emphasis shadow-none hover:cursor-pointer hover:bg-surface-hover hover:text-zinc-950 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-violet-500",
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
