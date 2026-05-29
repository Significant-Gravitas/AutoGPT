"use client";

import { cn } from "@/lib/utils";
import * as React from "react";
import { useScrollableTabs } from "../context";

interface Props extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  value: string;
}

export const ScrollableTabsTrigger = React.forwardRef<HTMLButtonElement, Props>(
  function ScrollableTabsTrigger(
    { className, value, children, ...props },
    ref,
  ) {
    const { activeValue, scrollToSection } = useScrollableTabs();
    const elementRef = React.useRef<HTMLButtonElement>(null);
    const isActive = activeValue === value;

    function handleClick(e: React.MouseEvent<HTMLButtonElement>) {
      e.preventDefault();
      e.stopPropagation();
      scrollToSection(value);
      props.onClick?.(e);
    }

    return (
      <button
        type="button"
        ref={(node) => {
          if (typeof ref === "function") ref(node);
          else if (ref) ref.current = node;
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          elementRef.current = node;
        }}
        data-scrollable-tab-trigger
        data-value={value}
        onClick={handleClick}
        className={cn(
          "relative inline-flex items-center justify-center whitespace-nowrap px-3 py-3 font-sans text-[0.875rem] font-medium leading-[1.5rem] text-zinc-700 transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
          isActive && "text-purple-600",
          className,
        )}
        {...props}
      >
        {children}
      </button>
    );
  },
);

ScrollableTabsTrigger.displayName = "ScrollableTabsTrigger";
