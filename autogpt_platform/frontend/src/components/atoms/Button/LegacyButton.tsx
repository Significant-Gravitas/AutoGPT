import { cn } from "@/lib/utils";
import { LegacyButtonProps, legacyButtonVariants } from "./helpers";

export function LegacyButton({
  className,
  variant,
  size,
  as: _as,
  ...props
}: LegacyButtonProps) {
  return (
    <button
      className={cn(legacyButtonVariants({ variant, size, className }))}
      {...props}
    />
  );
}
