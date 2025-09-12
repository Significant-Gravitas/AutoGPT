// BLOCK MENU TODO: We need a disable state in this, currently it's not in design.

import { cn } from "@/lib/utils";
import React from "react";

interface Props extends React.HTMLAttributes<HTMLDivElement> {
  selected?: boolean;
  children?: React.ReactNode; // For icon purpose
  disabled?: boolean;
}

export const ControlPanelButton: React.FC<Props> = ({
  selected = false,
  children,
  disabled,
  className,
  ...rest
}) => {
  return (
    // Using div instead of button, because it's only for design purposes. We are using this to give design to PopoverTrigger.
    <div
      role="button"
      className={cn(
        "flex h-[4.25rem] w-[4.25rem] items-center justify-center whitespace-normal bg-white p-[1.38rem] text-zinc-800 shadow-none hover:cursor-pointer hover:bg-zinc-100 hover:text-zinc-950 focus:ring-0",
        selected &&
          "bg-violet-50 text-violet-700 hover:cursor-default hover:bg-violet-50 hover:text-violet-700 active:bg-violet-50 active:text-violet-700",
        disabled && "cursor-not-allowed",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
};
