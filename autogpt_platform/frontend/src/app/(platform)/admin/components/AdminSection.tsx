"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ReactNode, useId } from "react";

interface Props {
  title?: string;
  description?: string;
  className?: string;
  children: ReactNode;
}

export function AdminSection({
  title,
  description,
  className,
  children,
}: Props) {
  const titleId = useId();

  return (
    <section
      aria-labelledby={title ? titleId : undefined}
      className={cn(
        "rounded-[18px] border border-zinc-200 bg-white px-5 py-4 shadow-[0_1px_2px_rgba(15,15,20,0.04)]",
        className,
      )}
    >
      {title ? (
        <div className="mb-4 flex flex-col">
          <Text variant="body-medium" as="h2" id={titleId}>
            {title}
          </Text>
          {description ? (
            <Text variant="small" className="text-zinc-500">
              {description}
            </Text>
          ) : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}
