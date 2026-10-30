"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";

function InputGroup({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="input-group"
      role="group"
      className={cn(
        "group/input-group relative flex w-full items-center rounded-xlarge border border-border-base bg-surface shadow-sm transition-[color,box-shadow] outline-none",
        "min-w-0 has-[>textarea]:h-auto",

        // Variants based on alignment.
        "has-[>[data-align=block-start]]:h-auto has-[>[data-align=block-start]]:flex-col",
        "has-[>[data-align=block-end]]:h-auto has-[>[data-align=block-end]]:flex-col",

        // Focus state.
        "has-[[data-slot=input-group-control]:focus-visible]:border-border-strong has-[[data-slot=input-group-control]:focus-visible]:ring-1 has-[[data-slot=input-group-control]:focus-visible]:ring-border-strong",

        className,
      )}
      {...props}
    />
  );
}

const inputGroupAddonVariants = cva(
  "text-muted-foreground flex h-auto cursor-text items-center justify-center gap-2 py-1.5 text-body font-medium leading-5 select-none group-data-[disabled=true]/input-group:opacity-50",
  {
    variants: {
      align: {
        "inline-start": "order-first pl-3",
        "inline-end": "order-last pr-3",
        "block-start": "order-first w-full justify-start px-3 pt-3",
        "block-end": "order-last w-full justify-start px-3 pb-3",
      },
    },
    defaultVariants: {
      align: "inline-start",
    },
  },
);

function InputGroupAddon({
  className,
  align = "inline-start",
  onClick,
  ...props
}: React.ComponentProps<"div"> & VariantProps<typeof inputGroupAddonVariants>) {
  return (
    <div
      role="group"
      data-slot="input-group-addon"
      data-align={align}
      className={cn(inputGroupAddonVariants({ align }), className)}
      onClick={(e) => {
        onClick?.(e);
        if (e.defaultPrevented) return;
        if ((e.target as HTMLElement).closest("button")) {
          return;
        }
        e.currentTarget.parentElement?.querySelector("textarea")?.focus();
      }}
      {...props}
    />
  );
}

// Callers name the looks of the deleted ui/button by their old names; this map
// keeps that API on the Button atom's compact family.
const INPUT_GROUP_BUTTON_VARIANT = {
  default: "compact-primary",
  ghost: "compact-ghost",
} as const;

function InputGroupButton({
  className,
  type = "button",
  variant = "ghost",
  size = "xs",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  size?: "xs" | "sm" | "icon-xs" | "icon-sm" | null;
  variant?: keyof typeof INPUT_GROUP_BUTTON_VARIANT;
}) {
  return (
    <Button
      type={type}
      data-size={size}
      variant={INPUT_GROUP_BUTTON_VARIANT[variant]}
      size="compact"
      className={cn(
        "flex min-w-0 items-center gap-2 text-body leading-5 shadow-none",
        size === "xs" && "h-6 gap-1 rounded-md px-2 has-[>svg]:px-2",
        size === "sm" && "h-8 gap-1.5 rounded-md px-2.5 has-[>svg]:px-2.5",
        size === "icon-xs" && "size-6 rounded-md p-0 has-[>svg]:p-0",
        size === "icon-sm" && "size-8 rounded-md p-0 has-[>svg]:p-0",
        className,
      )}
      {...props}
    />
  );
}

const InputGroupTextarea = React.forwardRef<
  HTMLTextAreaElement,
  React.ComponentProps<"textarea">
>(({ className, ...props }, ref) => {
  return (
    <Input
      bare
      type="textarea"
      ref={ref}
      data-slot="input-group-control"
      className={cn(
        "flex-1 resize-none rounded-none border-0 bg-transparent py-3 shadow-none focus-visible:ring-0",
        className,
      )}
      {...props}
    />
  );
});
InputGroupTextarea.displayName = "InputGroupTextarea";

export { InputGroup, InputGroupAddon, InputGroupButton, InputGroupTextarea };
