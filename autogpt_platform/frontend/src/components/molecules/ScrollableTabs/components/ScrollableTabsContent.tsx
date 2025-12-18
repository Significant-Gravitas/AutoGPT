"use client";

import { cn } from "@/lib/utils";
import * as React from "react";
import { useScrollableTabs } from "../context";

interface Props extends React.HTMLAttributes<HTMLDivElement> {
  value: string;
}

export const ScrollableTabsContent = React.forwardRef<HTMLDivElement, Props>(
  function ScrollableTabsContent(
    { className, value, children, ...props },
    ref,
  ) {
    const { registerContent } = useScrollableTabs();
    const contentRef = React.useRef<HTMLDivElement>(null);

    React.useEffect(() => {
      if (contentRef.current) {
        registerContent(value, contentRef.current);
      }
      return () => {
        registerContent(value, null);
      };
    }, [value, registerContent]);

    return (
      <div
        ref={(node) => {
          if (typeof ref === "function") ref(node);
          else if (ref) ref.current = node;
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          contentRef.current = node;
        }}
        data-scrollable-tab-content
        data-value={value}
        className={cn("focus-visible:outline-none", className)}
        {...props}
      >
        {children}
      </div>
    );
  },
);

ScrollableTabsContent.displayName = "ScrollableTabsContent";
