"use client";

import { cn } from "@/lib/utils";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import * as React from "react";

interface TabsLineContextValue {
  activeTabElement: HTMLElement | null;
  setActiveTabElement: React.Dispatch<React.SetStateAction<HTMLElement | null>>;
}

const TabsLineContext = React.createContext<TabsLineContextValue | undefined>(
  undefined,
);

function useTabsLine() {
  const context = React.useContext(TabsLineContext);
  if (!context) {
    throw new Error("useTabsLine must be used within a TabsLine");
  }
  return context;
}

function TabsLine(
  props: React.ComponentPropsWithoutRef<typeof TabsPrimitive.Root>,
) {
  const [activeTabElement, setActiveTabElement] =
    React.useState<HTMLElement | null>(null);

  return (
    <TabsLineContext.Provider value={{ activeTabElement, setActiveTabElement }}>
      <TabsPrimitive.Root {...props} />
    </TabsLineContext.Provider>
  );
}

const TabsLineList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => {
  const { activeTabElement } = useTabsLine();
  const listRef = React.useRef<HTMLDivElement>(null);

  return (
    <div className="relative">
      <TabsPrimitive.List
        ref={(node) => {
          if (typeof ref === "function") ref(node);
          else if (ref) ref.current = node;
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          listRef.current = node;
        }}
        className={cn(
          "inline-flex w-full items-center justify-start border-b border-zinc-100",
          className,
        )}
        {...props}
      />
      {activeTabElement && (
        <div
          className="transition-left transition-right absolute bottom-0 h-0.5 bg-purple-600 duration-200 ease-in-out"
          style={{
            left: activeTabElement.offsetLeft,
            width: activeTabElement.offsetWidth,
            willChange: "left, width",
          }}
        />
      )}
    </div>
  );
});
TabsLineList.displayName = "TabsLineList";

const TabsLineTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => {
  const elementRef = React.useRef<HTMLButtonElement>(null);
  const { setActiveTabElement } = useTabsLine();

  React.useEffect(() => {
    if (!elementRef.current) return;

    const observer = new MutationObserver(() => {
      if (!elementRef.current) return;
      if (elementRef.current.getAttribute("data-state") === "active") {
        setActiveTabElement(elementRef.current);
      }
    });

    observer.observe(elementRef.current, { attributes: true });

    // Initial check
    if (elementRef.current.getAttribute("data-state") === "active") {
      setActiveTabElement(elementRef.current);
    }

    return () => observer.disconnect();
  }, [setActiveTabElement]);

  return (
    <TabsPrimitive.Trigger
      ref={(node) => {
        if (typeof ref === "function") ref(node);
        else if (ref) ref.current = node;
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        elementRef.current = node;
      }}
      className={cn(
        "relative inline-flex items-center justify-center whitespace-nowrap px-3 py-3 font-sans text-[0.875rem] font-medium leading-[1.5rem] text-zinc-700 transition-all data-[state=active]:text-purple-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
});
TabsLineTrigger.displayName = "TabsLineTrigger";

const TabsLineContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn(
      "mt-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-stone-400 focus-visible:ring-offset-2",
      className,
    )}
    {...props}
  />
));
TabsLineContent.displayName = "TabsLineContent";

export { TabsLine, TabsLineContent, TabsLineList, TabsLineTrigger };
