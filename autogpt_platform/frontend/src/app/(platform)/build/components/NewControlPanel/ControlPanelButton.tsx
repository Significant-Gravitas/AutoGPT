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
          "flex w-auto items-center justify-center whitespace-normal bg-white px-4 py-4 text-zinc-800 shadow-none hover:cursor-pointer hover:bg-zinc-100 hover:text-zinc-950 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-violet-500",
          selected &&
            "bg-violet-50 text-violet-700 hover:cursor-default hover:bg-violet-50 hover:text-violet-700 active:bg-violet-50 active:text-violet-700",
          disabled && "cursor-not-allowed opacity-50 hover:cursor-not-allowed",
          className,
        )}
        {...rest}
      />
    );
  },
);
