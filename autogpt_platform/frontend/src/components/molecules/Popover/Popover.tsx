"use client";

import {
  Popover,
  PopoverContent as KobraPopoverContent,
  PopoverTrigger as KobraPopoverTrigger,
} from "@/components/ui/popover";
import { Popover as PopoverPrimitive } from "@base-ui/react/popover";
import * as React from "react";

type AsChildProps<P> = P & {
  /** Radix form: the single child becomes the rendered element. */
  asChild?: boolean;
};

function renderFromChild<P extends { children?: React.ReactNode }>({
  asChild,
  children,
  ...props
}: AsChildProps<P>) {
  if (asChild && React.isValidElement(children)) {
    return { ...props, render: children };
  }
  return { ...props, children };
}

function PopoverTrigger(
  props: AsChildProps<React.ComponentProps<typeof KobraPopoverTrigger>>,
) {
  return <KobraPopoverTrigger {...renderFromChild(props)} />;
}

type ContentProps = AsChildProps<
  React.ComponentProps<typeof KobraPopoverContent>
> & {
  /** Radix collision padding; Base UI positions with its own default. */
  collisionPadding?: number;
  /** Radix hook; Base UI focuses the popup itself. Use `initialFocus`. */
  onOpenAutoFocus?: (event: Event) => void;
};

function PopoverContent({
  collisionPadding: _collisionPadding,
  onOpenAutoFocus: _onOpenAutoFocus,
  ...props
}: ContentProps) {
  return <KobraPopoverContent {...renderFromChild(props)} />;
}

// Kobra's popover has no Close part; Base UI's closes the nearest Popover.
function PopoverClose(props: PopoverPrimitive.Close.Props) {
  return <PopoverPrimitive.Close data-slot="popover-close" {...props} />;
}

export { Popover, PopoverClose, PopoverContent, PopoverTrigger };
