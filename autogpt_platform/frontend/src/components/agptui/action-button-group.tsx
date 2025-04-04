import React from "react";
import { cn } from "@/lib/utils";

import type { ButtonAction } from "@/components/agptui/types";
import { Button, buttonVariants } from "@/components/agptui/Button";
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
            onClick={action.callback}
          >
            {action.label}
          </Button>
        ) : (
          <Link
            key={i}
            className={buttonVariants({ variant: action.variant })}
            href={action.href}
          >
            {action.label}
          </Link>
        ),
      )}
    </div>
  );
}
