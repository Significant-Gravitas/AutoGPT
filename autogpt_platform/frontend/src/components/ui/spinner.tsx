import { cn } from "@/lib/utils";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

function Spinner({ className, ...props }: React.ComponentProps<"svg">) {
  return (
    <Icon
      icon={Loading03Icon}
      role="status"
      aria-label="Loading"
      className={cn("size-4 animate-spin", className)}
      {...(props as Record<string, unknown>)}
    />
  );
}

export { Spinner };
