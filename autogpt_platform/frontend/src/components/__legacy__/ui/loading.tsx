import { cn } from "@/lib/utils";
import { LoaderCircle } from "lucide-react";

export default function LoadingBox({
  className,
  spinnerSize,
}: {
  className?: string;
  spinnerSize?: string | number;
}) {
  const spinnerSizeClass =
    typeof spinnerSize == "string"
      ? `size-[${spinnerSize}]`
      : typeof spinnerSize == "number"
        ? `size-${spinnerSize}`
        : undefined;
  return (
    <div className={cn("flex items-center justify-center", className)}>
      <LoadingSpinner className={spinnerSizeClass} />
    </div>
  );
}

export function LoadingSpinner({ className }: { className?: string }) {
  return <LoaderCircle className={cn("size-16 animate-spin", className)} />;
}
