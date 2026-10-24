import { Cancel01Icon } from "@hugeicons/core-free-icons";
import type { ComponentPropsWithoutRef } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";

interface DismissProps
  extends Omit<ComponentPropsWithoutRef<"button">, "children"> {
  "aria-label": string;
  onClick: () => void;
}

export function Dismiss({ className, ...rest }: DismissProps) {
  return (
    <Button
      data-slot="attention-card-dismiss"
      type="button"
      variant="icon"
      size="icon-xs"
      className={className}
      {...rest}
    >
      <Icon icon={Cancel01Icon} className="h-3.5 w-3.5" />
    </Button>
  );
}
