import type { ReactNode } from "react";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  title: string;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function OverviewCard({ title, action, className, children }: Props) {
  return (
    <section
      aria-label={title}
      className={cn(
        "flex min-w-0 flex-col gap-3 rounded-2xl bg-white p-4 smooth-shadow-ring-sm",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <Text variant="body-medium" tone="primary">
          {title}
        </Text>
        {action}
      </div>
      {children}
    </section>
  );
}
