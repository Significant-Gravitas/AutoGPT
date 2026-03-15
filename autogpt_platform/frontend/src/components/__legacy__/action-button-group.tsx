import React from "react";
import { cn } from "@/lib/utils";

import type { ButtonAction } from "@/components/__legacy__/types";
import { Button, buttonVariants } from "@/components/__legacy__/Button";
import Link from "next/link";

export default function ActionButtonGroup({
  title,
  actions,
  className,
}: {
  title: React.ReactNode;
  actions: ButtonAction[];
  className?: string;
}): React.ReactElement {
  return (
    <div className={cn("flex flex-col gap-3", className)}>
      <h3 className="text-sm font-medium">{title}</h3>
      {actions.map((action, i) =>
        "callback" in action ? (
          <Button
            key={i}
            variant={action.variant ?? "outline"}
            disabled={action.disabled}
            onClick={action.callback}
            {...action.extraProps}
          >
            {action.label}
          </Button>
        ) : (
          <Link
            key={i}
            className={cn(
              buttonVariants({ variant: action.variant }),
              action.disabled &&
                "pointer-events-none border-zinc-400 text-zinc-400",
            )}
            href={action.href}
            {...action.extraProps}
          >
            {action.label}
          </Link>
        ),
      )}
    </div>
  );
}
