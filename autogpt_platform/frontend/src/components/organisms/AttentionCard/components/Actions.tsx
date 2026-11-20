import type { VariantProps } from "class-variance-authority";
import type { ComponentPropsWithoutRef } from "react";
import { cn } from "@/lib/utils";
import { actionsVariants } from "../helpers";

interface ActionsProps
  extends ComponentPropsWithoutRef<"div">,
    VariantProps<typeof actionsVariants> {}

export function Actions({ placement, className, ...rest }: ActionsProps) {
  return (
    <div
      data-slot="attention-card-actions"
      className={cn(actionsVariants({ placement }), className)}
      {...rest}
    />
  );
}
