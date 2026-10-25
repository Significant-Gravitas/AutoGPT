import { cn } from "@/lib/utils";
import React from "react";

interface Props extends React.HTMLAttributes<HTMLElement> {
  selected?: boolean;
  children?: React.ReactNode;
  disabled?: boolean;
  as?: "div" | "button";
}

export const ControlPanelButton: React.FC<Props> = ({
  selected = false,
  children,
  disabled,
  as = "div",
  className,
  ...rest
}) => {
  const Component = as;

  return (
    // Why div - because on some places we are only using this for design purposes.
    <Component
      role={as === "div" ? "button" : undefined}
      disabled={as === "button" ? disabled : undefined}
      className={cn(
        // TODO(tokens): no fitting role for hover:text-zinc-950 #09090B; #09090B is foreground, whose .dark value (250 250 250) differs (QUALITY 43, SECRT-3154)
        "flex w-auto items-center justify-center whitespace-normal bg-surface px-4 py-4 text-foreground-emphasis shadow-none hover:cursor-pointer hover:bg-surface-hover hover:text-zinc-950 focus:ring-0",
        selected &&
          "bg-accent-surface text-accent-strong hover:cursor-default hover:bg-accent-surface hover:text-accent-strong active:bg-accent-surface active:text-accent-strong",
        disabled && "cursor-not-allowed opacity-50 hover:cursor-not-allowed",
        className,
      )}
      {...rest}
    >
      {children}
    </Component>
  );
};
