"use client";

import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import {
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import * as React from "react";

const contentClassName = "rounded-xl";

const itemClassName = "cursor-pointer px-3 py-2";

type Variant = "default" | "destructive";

const SecondaryMenu = ContextMenu;
const SecondaryMenuTrigger = ContextMenuTrigger;

function SecondaryMenuContent({
  className,
  ...props
}: React.ComponentProps<typeof ContextMenuContent>) {
  return (
    <ContextMenuContent
      className={cn(contentClassName, className)}
      {...props}
    />
  );
}

interface SecondaryMenuItemProps extends Omit<
  React.ComponentProps<typeof ContextMenuItem>,
  "onSelect"
> {
  variant?: Variant;
  /** Radix name for the activation handler; Base UI calls it `onClick`.
   *  `preventDefault()` keeps the menu open, as it did there. */
  onSelect?: (event: React.MouseEvent<HTMLDivElement>) => void;
}

function SecondaryMenuItem({
  className,
  onSelect,
  onClick,
  ...props
}: SecondaryMenuItemProps) {
  return (
    <ContextMenuItem
      className={cn(itemClassName, className)}
      onClick={(event) => {
        onClick?.(event);
        if (!onSelect) return;
        onSelect(event);
        if (event.defaultPrevented) event.preventBaseUIHandler();
      }}
      {...props}
    />
  );
}

const SecondaryMenuSeparator = ContextMenuSeparator;

function SecondaryDropdownMenuContent({
  className,
  ...props
}: React.ComponentProps<typeof DropdownMenuContent>) {
  return (
    <DropdownMenuContent
      className={cn(contentClassName, "w-auto", className)}
      {...props}
    />
  );
}

function SecondaryDropdownMenuItem({
  className,
  ...props
}: React.ComponentProps<typeof DropdownMenuItem>) {
  return (
    <DropdownMenuItem className={cn(itemClassName, className)} {...props} />
  );
}

const SecondaryDropdownMenuSeparator = DropdownMenuSeparator;

export {
  SecondaryMenu,
  SecondaryMenuTrigger,
  SecondaryMenuContent,
  SecondaryMenuItem,
  SecondaryMenuSeparator,
  SecondaryDropdownMenuContent,
  SecondaryDropdownMenuItem,
  SecondaryDropdownMenuSeparator,
};
