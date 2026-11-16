import type { ComponentPropsWithoutRef } from "react";
import { cn } from "@/lib/utils";

// A decorative hairline: it is drawn, not announced, like the border it
// replaces.
export function Separator({
  className,
  ...rest
}: ComponentPropsWithoutRef<"div">) {
  return (
    <div
      data-slot="attention-card-separator"
      aria-hidden
      className={cn("border-t border-separator", className)}
      {...rest}
    />
  );
}
