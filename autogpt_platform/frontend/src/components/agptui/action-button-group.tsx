import React from "react";
import { cn } from "@/lib/utils";

import type { ButtonAction } from "@/components/agptui/types";
import { Button, buttonVariants } from "@/components/agptui/Button";
import Link from "next/link";

export default function ActionButtonGroup({
  actions,
  className,
  title,
}: {
  actions: ButtonAction[];
  className?: string;
  title: React.ReactNode;
}): React.ReactElement {
  return (
    <div className={cn(className, "flex flex-col gap-3")}>
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
