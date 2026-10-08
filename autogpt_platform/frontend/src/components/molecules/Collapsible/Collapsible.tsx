"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Collapsible as KobraCollapsible,
  CollapsibleContent as KobraCollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import type { ComponentProps, ReactNode } from "react";

type WithClassName<T> = Omit<T, "className"> & { className?: string };

type RootProps = WithClassName<ComponentProps<typeof KobraCollapsible>>;

interface ComposedProps {
  trigger: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  className?: string;
  triggerClassName?: string;
  contentClassName?: string;
}

/**
 * With `trigger` it renders the house header-plus-chevron layout; without it,
 * it is the bare root to compose with `CollapsibleTrigger` and
 * `CollapsibleContent`.
 */
export function Collapsible(props: RootProps | ComposedProps) {
  if (!("trigger" in props)) {
    return <KobraCollapsible {...props} />;
  }

  const {
    trigger,
    children,
    defaultOpen = false,
    open,
    onOpenChange,
    className,
    triggerClassName,
    contentClassName,
  } = props;

  return (
    <KobraCollapsible
      open={open}
      defaultOpen={defaultOpen}
      onOpenChange={(next) => onOpenChange?.(next)}
      className={cn("w-full", className)}
    >
      <CollapsibleTrigger
        className={cn(
          "group/collapsible-trigger flex w-full items-center justify-between text-left transition-all duration-200 hover:opacity-80",
          triggerClassName,
        )}
      >
        <div className="flex flex-wrap items-center gap-2">
          {trigger}
          <Icon
            icon={ArrowDown01Icon}
            className="inline-flex h-4 w-4 transition-transform duration-200 group-data-panel-open/collapsible-trigger:rotate-180"
          />
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent className={contentClassName}>
        <div className="pt-2">{children}</div>
      </CollapsibleContent>
    </KobraCollapsible>
  );
}

export function CollapsibleContent({
  className,
  ...props
}: WithClassName<ComponentProps<typeof KobraCollapsibleContent>>) {
  return (
    <KobraCollapsibleContent
      className={cn(
        "h-(--collapsible-panel-height) overflow-hidden transition-[height] duration-200 ease-out data-ending-style:h-0 data-starting-style:h-0 motion-reduce:transition-none",
        className,
      )}
      {...props}
    />
  );
}

export { CollapsibleTrigger };
