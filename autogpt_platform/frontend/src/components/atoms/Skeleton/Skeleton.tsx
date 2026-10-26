import { cn } from "@/lib/utils";

interface Props extends React.HTMLAttributes<HTMLDivElement> {
  className?: string;
}

export function Skeleton({ className, ...props }: Props) {
  return (
    <div
      data-slot="skeleton"
      className={cn("animate-pulse rounded-md bg-surface-hover", className)}
      {...props}
    />
  );
}
