"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import type { IconSvgElement } from "@hugeicons/react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import * as React from "react";

type TabsLineVariant = "default" | "compact" | "pill";

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
   * `pill` is a segmented control: triggers in a rounded grey track, the
   * active one raised on a light chip, and no underline.
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
      <TabsPrimitive.Root data-slot="tabs" {...props} />
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

  if (variant === "pill") {
    return (
      <TabsPrimitive.List
        ref={ref}
        data-slot="tabs-list"
        // TODO(tokens): no role holds gray-100 (#F3F4F6) or neutral-500
        // (#737373). The dark: classes never apply (forcedTheme="light") and
        // are DS6's (SECRT-3116) to delete.
        className={cn(
          "inline-flex h-10 w-80 items-center justify-center rounded-3xl bg-gray-100 p-[5px] text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400",
          className,
        )}
        {...props}
      />
    );
  }

  return (
    <div data-slot="tabs-list-root" className="relative">
      <TabsPrimitive.List
        ref={(node) => {
          if (typeof ref === "function") ref(node);
          else if (ref) ref.current = node;
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          listRef.current = node;
        }}
        data-slot="tabs-list"
        className={cn(
          "inline-flex w-full items-center justify-start border-b border-separator",
          isFlush && "[&>button:first-child]:!pl-0",
          className,
        )}
        {...props}
      />
      {activeTabElement && (
        <div
          data-slot="tabs-indicator"
          className={cn(
            "transition-left transition-right absolute bottom-0 h-0.5 bg-brand-hover duration-200 ease-in-out",
            isCompact && "bg-action-hover",
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
      data-slot="tabs-trigger"
      // TODO(tokens): no role for neutral-400 (#A3A3A3), and in `pill` no
      // role holds gray-500 (#6B7280).
      className={cn(
        variant === "pill" && [
          "flex flex-1 items-start justify-center gap-2.5 whitespace-nowrap rounded-2xl px-3 py-2 text-center font-sans text-caption font-medium leading-tight text-gray-500 ring-offset-surface transition-all",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
          "data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-subtle",
        ],
        variant !== "pill" &&
          "relative inline-flex items-center justify-center whitespace-nowrap px-3 py-3 font-sans text-body font-medium leading-6 text-foreground-body transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:text-brand-hover",
        icon && variant !== "pill" && "gap-1.5",
        variant === "compact" &&
          "px-2.5 py-2 text-body leading-5 text-foreground-secondary data-[state=active]:text-foreground-strong",
        className,
      )}
      {...props}
    >
      {icon ? (
        <Icon data-slot="tabs-trigger-icon" icon={icon} size={14} aria-hidden />
      ) : null}
      {children}
    </TabsPrimitive.Trigger>
  );
});
TabsLineTrigger.displayName = "TabsLineTrigger";

const TabsLineContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => {
  const isPill = React.useContext(TabsLineContext)?.variant === "pill";

  return (
    <TabsPrimitive.Content
      ref={ref}
      data-slot="tabs-panel"
      // TODO(tokens): no role for stone-400 (#A8A29E). The dark: classes in
      // `pill` never apply (forcedTheme="light") and are DS6's (SECRT-3116) to
      // delete.
      // Radix marks inactive panels with the `hidden` attribute, but the UA
      // rule behind it is weaker than any author display utility — a panel
      // styled `flex`/`grid` stays laid out and keeps stealing space from the
      // active one. The data-state variant is specific enough to win. `pill`
      // panels keep the UA rule, as the segmented control always has.
      className={cn(
        isPill
          ? "mt-2 ring-offset-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 dark:ring-offset-neutral-950 dark:focus-visible:ring-neutral-300"
          : "mt-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-stone-400 focus-visible:ring-offset-2 data-[state=inactive]:hidden",
        className,
      )}
      {...props}
    />
  );
});
TabsLineContent.displayName = "TabsLineContent";

export { TabsLine, TabsLineContent, TabsLineList, TabsLineTrigger };
