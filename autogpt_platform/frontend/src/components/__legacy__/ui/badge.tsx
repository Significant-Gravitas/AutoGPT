import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border border-zinc-200 px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-hidden focus:ring-2 focus:ring-black focus:ring-offset-2 cursor-default",
  {
    variants: {
      variant: {
        default: "border-transparent bg-zinc-900 text-zinc-50 shadow-sm",
        secondary: "border-transparent bg-zinc-100 text-zinc-900",
        destructive: "border-transparent bg-red-500 text-zinc-50 shadow-sm",
        outline: "text-black",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface BadgeProps
  extends
    React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
