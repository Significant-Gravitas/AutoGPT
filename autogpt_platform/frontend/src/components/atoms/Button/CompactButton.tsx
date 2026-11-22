import { cn } from "@/lib/utils";
import { CompactButtonProps, compactButtonVariants } from "./compact-variants";

// Renders its children bare, as the deleted components/ui/button did, so the
// call sites' DOM stays the same.
export function CompactButton({
  className,
  variant,
  size,
  disabled,
  as: _as,
  ...props
}: CompactButtonProps) {
  return (
    <button
      data-slot="button"
      className={cn(compactButtonVariants({ variant, size, className }))}
      disabled={disabled}
      {...props}
    />
  );
}
