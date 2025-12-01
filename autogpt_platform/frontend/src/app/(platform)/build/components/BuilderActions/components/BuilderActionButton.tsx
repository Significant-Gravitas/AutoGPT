import { Button } from "@/components/atoms/Button/Button";
import { ButtonProps } from "@/components/atoms/Button/helpers";
import { cn } from "@/lib/utils";
import { CircleNotchIcon } from "@phosphor-icons/react";

export const BuilderActionButton = ({
  children,
  className,
  isLoading,
  ...props
}: ButtonProps & { isLoading?: boolean }) => {
  return (
    <Button
      variant="icon"
      size={"small"}
      className={cn(
        "relative h-12 w-12 min-w-0 text-lg",
        "bg-gradient-to-br from-zinc-50 to-zinc-200",
        "border border-zinc-200",
        "shadow-[inset_0_3px_0_0_rgba(255,255,255,0.5),0_2px_4px_0_rgba(0,0,0,0.2)]",
        "dark:shadow-[inset_0_1px_0_0_rgba(255,255,255,0.1),0_2px_4px_0_rgba(0,0,0,0.4)]",
        "hover:shadow-[inset_0_1px_0_0_rgba(255,255,255,0.5),0_1px_2px_0_rgba(0,0,0,0.2)]",
        "active:shadow-[inset_0_2px_4px_0_rgba(0,0,0,0.2)]",
        "transition-all duration-150",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    >
      {!isLoading ? (
        children
      ) : (
        <CircleNotchIcon className="size-6 animate-spin" />
      )}
    </Button>
  );
};
