import type { IconSvgElement } from "@hugeicons/react";
import type { ReactNode } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  children: ReactNode;
  icon: IconSvgElement;
  title: string;
  /** Sits right after the title: a count, a status. */
  badge?: ReactNode;
  /** Right-aligned header content: a filter, a summary line. */
  meta?: ReactNode;
  className?: string;
  contentClassName?: string;
  as?: "div" | "section";
}

/** A flat panel with a one-line header row. Rows inside carry their own
 *  horizontal padding so dividers can run edge to edge. */
export function HomeTile({
  children,
  icon,
  title,
  badge,
  meta,
  className,
  contentClassName,
  as = "section",
}: Props) {
  const Component = as;
  return (
    <Component
      className={cn(
        "flex min-w-0 flex-col overflow-hidden rounded-lg border border-zinc-200 bg-white",
        className,
      )}
    >
      <div className="flex min-h-10 items-center justify-between gap-3 border-b border-zinc-100 px-4">
        <div className="flex min-w-0 items-center gap-2">
          <Icon
            icon={icon}
            size={15}
            className="shrink-0 text-zinc-400"
            aria-hidden="true"
          />
          <Text
            variant="body-medium"
            as="h2"
            className="truncate text-sm text-zinc-900"
          >
            {title}
          </Text>
          {badge}
        </div>
        {meta ? (
          <div className="flex shrink-0 items-center gap-2 text-xs text-zinc-500">
            {meta}
          </div>
        ) : null}
      </div>
      <div className={cn("flex min-w-0 flex-1 flex-col", contentClassName)}>
        {children}
      </div>
    </Component>
  );
}
