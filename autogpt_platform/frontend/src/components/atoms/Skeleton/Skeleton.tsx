import { cn } from "@/lib/utils";

interface Props extends React.HTMLAttributes<HTMLDivElement> {
  className?: string;
}

export function Skeleton({ className, ...props }: Props) {
  return (
    <div
      data-slot="skeleton"
      // TODO(tokens): no fitting role for bg-zinc-100 #EFEFF0; #EFEFF0 is
      // surface-hover, a hover fill, and this fill is not a hover state
      // (QUALITY 41, SECRT-3154).
      className={cn("animate-pulse rounded-md bg-zinc-100", className)}
      {...props}
    />
  );
}
