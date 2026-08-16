import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props {
  children: ReactNode;
  title: ReactNode;
  header?: ReactNode;
  className?: string;
  contentClassName?: string;
  surfaceClassName?: string;
  as?: "div" | "section";
}

export function HomeTile({
  children,
  title,
  header,
  className,
  contentClassName,
  surfaceClassName,
  as = "section",
}: Props) {
  const Component = as;
  return (
    <Component className={cn("relative flex min-w-0 flex-col", className)}>
      <div className="mb-2 flex flex-col justify-start px-4 sm:px-5">
        {title}
        {header ? <div className="mt-0.5">{header}</div> : null}
      </div>
      <div
        className={cn(
          "relative flex min-w-0 flex-1 flex-col rounded-[30px] bg-white p-4 shadow-zinc-950 smooth-shadow-ring-sm sm:p-5",
          surfaceClassName,
        )}
      >
        <div className={cn("min-w-0 flex-1", contentClassName)}>{children}</div>
      </div>
    </Component>
  );
}
