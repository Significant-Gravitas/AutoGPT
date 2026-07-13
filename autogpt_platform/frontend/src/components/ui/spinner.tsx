import { CircleNotch as CircleNotchIcon } from "@/components/icons/pika/adapter";

import { cn } from "@/lib/utils";

function Spinner({ className, ...props }: React.ComponentProps<"svg">) {
  return (
    <CircleNotchIcon
      role="status"
      aria-label="Loading"
      className={cn("size-4 animate-spin", className)}
      {...(props as Record<string, unknown>)}
    />
  );
}

export { Spinner };
