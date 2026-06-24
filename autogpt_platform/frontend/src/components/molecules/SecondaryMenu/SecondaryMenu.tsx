"use client";

import { cn } from "@/lib/utils";
import * as ContextMenu from "@radix-ui/react-context-menu";
import * as DropdownMenuPrimitive from "@radix-ui/react-dropdown-menu";
import React from "react";

const secondaryMenuContentClassName =
  "z-10 rounded-xl border bg-white p-1 shadow-md dark:bg-gray-800";

const secondaryMenuItemClassName =
  "flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700";

const secondaryMenuSeparatorClassName =
  "my-1 h-px bg-gray-300 dark:bg-gray-600";

export const SecondaryMenuContent = React.forwardRef<
  React.ElementRef<typeof ContextMenu.Content>,
  React.ComponentPropsWithoutRef<typeof ContextMenu.Content>
>(({ className, ...props }, ref) => (
  <ContextMenu.Content
    ref={ref}
    className={cn(secondaryMenuContentClassName, className)}
    {...props}
  />
));
SecondaryMenuContent.displayName = "SecondaryMenuContent";

export const SecondaryMenuItem = React.forwardRef<
  React.ElementRef<typeof ContextMenu.Item>,
  React.ComponentPropsWithoutRef<typeof ContextMenu.Item> & {
    variant?: "default" | "destructive";
  }
>(({ className, variant = "default", ...props }, ref) => (
  <ContextMenu.Item
    ref={ref}
    className={cn(
      secondaryMenuItemClassName,
      variant === "destructive" &&
        "text-red-500 hover:bg-gray-100 dark:hover:bg-gray-700",
      className,
    )}
    {...props}
  />
));
SecondaryMenuItem.displayName = "SecondaryMenuItem";

export const SecondaryMenuSeparator = React.forwardRef<
  React.ElementRef<typeof ContextMenu.Separator>,
  React.ComponentPropsWithoutRef<typeof ContextMenu.Separator>
>(({ className, ...props }, ref) => (
  <ContextMenu.Separator
    ref={ref}
    className={cn(secondaryMenuSeparatorClassName, className)}
    {...props}
  />
));
SecondaryMenuSeparator.displayName = "SecondaryMenuSeparator";

export const SecondaryDropdownMenuContent = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Content>
>(({ className, ...props }, ref) => (
  <DropdownMenuPrimitive.Portal>
    <DropdownMenuPrimitive.Content
      ref={ref}
      className={cn(secondaryMenuContentClassName, className)}
      {...props}
    />
  </DropdownMenuPrimitive.Portal>
));
SecondaryDropdownMenuContent.displayName = "SecondaryDropdownMenuContent";

export const SecondaryDropdownMenuItem = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Item> & {
    variant?: "default" | "destructive";
  }
>(({ className, variant = "default", ...props }, ref) => (
  <DropdownMenuPrimitive.Item
    ref={ref}
    className={cn(
      secondaryMenuItemClassName,
      variant === "destructive" &&
        "text-red-500 hover:bg-gray-100 dark:hover:bg-gray-700",
      className,
    )}
    {...props}
  />
));
SecondaryDropdownMenuItem.displayName = "SecondaryDropdownMenuItem";

export const SecondaryDropdownMenuSeparator = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <DropdownMenuPrimitive.Separator
    ref={ref}
    className={cn(secondaryMenuSeparatorClassName, className)}
    {...props}
  />
));
SecondaryDropdownMenuSeparator.displayName = "SecondaryDropdownMenuSeparator";
