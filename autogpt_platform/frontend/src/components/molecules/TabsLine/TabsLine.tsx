"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { Tabs as TabsPrimitive } from "@base-ui/react/tabs";
import type { IconSvgElement } from "@hugeicons/react";
import * as React from "react";

type TabsLineVariant = "default" | "compact";

const TabsLineContext = React.createContext<TabsLineVariant>("default");

type WithClassName<T> = Omit<T, "className"> & { className?: string };

interface TabsLineProps extends WithClassName<
  Omit<
    React.ComponentProps<typeof Tabs>,
    "value" | "defaultValue" | "onValueChange"
  >
> {
  /**
   * `compact` is the dense neutral style: flush, foreground underline, 14px
   * triggers with tighter padding. `default` underlines in the accent.
   */
  variant?: TabsLineVariant;
  value?: string;
  defaultValue?: string;
  onValueChange?: (value: string) => void;
}

function TabsLine({
  variant = "default",
  className,
  onValueChange,
  ...props
}: TabsLineProps) {
  return (
    <TabsLineContext value={variant}>
      <Tabs
        className={cn("gap-0", className)}
        onValueChange={(next) => {
          if (typeof next === "string") onValueChange?.(next);
        }}
        {...props}
      />
    </TabsLineContext>
  );
}

interface TabsLineListProps extends WithClassName<
  React.ComponentProps<typeof TabsList>
> {
  /**
   * When `true`, removes the left padding on the first tab trigger so it
   * aligns flush with the list's left edge. Defaults to `false`.
   */
  flush?: boolean;
  /**
   * Overrides the active-tab underline colour, for surfaces that want a
   * neutral underline instead of the accent.
   */
  indicatorClassName?: string;
}

function TabsLineList({
  className,
  flush,
  indicatorClassName,
  children,
  ...props
}: TabsLineListProps) {
  const variant = React.useContext(TabsLineContext);
  const isCompact = variant === "compact";
  const isFlush = flush ?? isCompact;

  return (
    <TabsList
      variant="line"
      className={cn(
        "relative w-full justify-start gap-0 rounded-none border-b border-border p-0 group-data-horizontal/tabs:h-auto",
        isFlush && "[&>button:first-child]:pl-0!",
        className,
      )}
      {...props}
    >
      {children}
      <TabsPrimitive.Indicator
        renderBeforeHydration
        className={cn(
          "absolute bottom-0 left-(--active-tab-left) h-0.5 w-(--active-tab-width) bg-accent transition-[left,width] duration-200 ease-in-out motion-reduce:transition-none",
          isCompact && "bg-foreground",
          indicatorClassName,
        )}
      />
    </TabsList>
  );
}

interface TabsLineTriggerProps extends WithClassName<
  React.ComponentProps<typeof TabsTrigger>
> {
  value: string;
  /** Hugeicon shown before the label at 14px. */
  icon?: IconSvgElement;
}

function TabsLineTrigger({
  className,
  icon,
  children,
  ...props
}: TabsLineTriggerProps) {
  const variant = React.useContext(TabsLineContext);

  return (
    <TabsTrigger
      className={cn(
        "h-auto flex-none rounded-none border-0 px-3 py-3 font-sans text-sm leading-6 font-medium text-muted-foreground after:hidden data-active:bg-transparent data-active:text-accent",
        variant === "compact" &&
          "px-2.5 py-2 leading-5 data-active:text-foreground",
        className,
      )}
      {...props}
    >
      {icon ? <Icon icon={icon} size={14} aria-hidden /> : null}
      {children}
    </TabsTrigger>
  );
}

function TabsLineContent({
  className,
  ...props
}: WithClassName<React.ComponentProps<typeof TabsContent>>) {
  return (
    <TabsContent
      className={cn("mt-4 focus-ring focus-visible:ring-offset-2", className)}
      {...props}
    />
  );
}

export { TabsLine, TabsLineContent, TabsLineList, TabsLineTrigger };
