"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import type { IconSvgElement } from "@hugeicons/react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import * as React from "react";

type TabsLineVariant = "default" | "compact";

interface TabsLineContextValue {
  variant: TabsLineVariant;
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

interface TabsLineProps
  extends React.ComponentPropsWithoutRef<typeof TabsPrimitive.Root> {
  /**
   * `compact` is the dense neutral style: flush, zinc underline, 14px
   * triggers with tighter padding. `default` keeps the purple accent.
   */
  variant?: TabsLineVariant;
}

function TabsLine({ variant = "default", ...props }: TabsLineProps) {
  const [activeTabElement, setActiveTabElement] =
    React.useState<HTMLElement | null>(null);

  return (
    <TabsLineContext.Provider
      value={{ variant, activeTabElement, setActiveTabElement }}
    >
      <TabsPrimitive.Root {...props} />
    </TabsLineContext.Provider>
  );
}

interface TabsLineListProps
  extends React.ComponentPropsWithoutRef<typeof TabsPrimitive.List> {
  /**
   * When `true`, removes the left padding on the first tab trigger so it
   * aligns flush with the list's left edge. Defaults to `false`.
   */
  flush?: boolean;
  /**
   * Overrides the active-tab underline colour, for surfaces that want a
   * neutral accent instead of the default purple.
   */
  indicatorClassName?: string;
}

const TabsLineList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  TabsLineListProps
>(({ className, flush, indicatorClassName, ...props }, ref) => {
  const { variant, activeTabElement } = useTabsLine();
  const listRef = React.useRef<HTMLDivElement>(null);
  const isCompact = variant === "compact";
  const isFlush = flush ?? isCompact;

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
          isFlush && "[&>button:first-child]:!pl-0",
          className,
        )}
        {...props}
      />
      {activeTabElement && (
        <div
          className={cn(
            "transition-left transition-right absolute bottom-0 h-0.5 bg-purple-600 duration-200 ease-in-out",
            isCompact && "bg-zinc-900",
            indicatorClassName,
          )}
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

interface TabsLineTriggerProps
  extends React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger> {
  /** Hugeicon shown before the label at 14px. */
  icon?: IconSvgElement;
}

const TabsLineTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  TabsLineTriggerProps
>(({ className, icon, children, ...props }, ref) => {
  const elementRef = React.useRef<HTMLButtonElement>(null);
  const { variant, setActiveTabElement } = useTabsLine();

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
        icon && "gap-1.5",
        variant === "compact" &&
          "px-2.5 py-2 text-sm leading-5 text-zinc-600 data-[state=active]:text-zinc-900",
        className,
      )}
      {...props}
    >
      {icon ? <Icon icon={icon} size={14} aria-hidden /> : null}
      {children}
    </TabsPrimitive.Trigger>
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
      // Radix marks inactive panels with the `hidden` attribute, but the UA
      // rule behind it is weaker than any author display utility — a panel
      // styled `flex`/`grid` stays laid out and keeps stealing space from the
      // active one. The data-state variant is specific enough to win.
      "mt-4 data-[state=inactive]:hidden focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-stone-400 focus-visible:ring-offset-2",
      className,
    )}
    {...props}
  />
));
TabsLineContent.displayName = "TabsLineContent";

export { TabsLine, TabsLineContent, TabsLineList, TabsLineTrigger };
