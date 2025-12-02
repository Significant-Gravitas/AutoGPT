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
        "flex h-[4.25rem] w-[4.25rem] items-center justify-center whitespace-normal bg-white p-[1.38rem] text-zinc-800 shadow-none hover:cursor-pointer hover:bg-zinc-100 hover:text-zinc-950 focus:ring-0",
        selected &&
          "bg-violet-50 text-violet-700 hover:cursor-default hover:bg-violet-50 hover:text-violet-700 active:bg-violet-50 active:text-violet-700",
        disabled && "cursor-not-allowed opacity-50 hover:cursor-not-allowed",
        className,
      )}
      {...rest}
    >
      {children}
    </Component>
  );
};
