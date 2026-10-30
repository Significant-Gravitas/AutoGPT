import { cn } from "@/lib/utils";
import { forwardRef } from "react";

export type BaseInputProps = React.InputHTMLAttributes<HTMLInputElement>;

// The bare <input> under the Input atom, for fields that bring their own label
// and layout.
export const BaseInput = forwardRef<HTMLInputElement, BaseInputProps>(
  function BaseInput({ className, type, ...props }, ref) {
    return (
      <input
        type={type}
        className={cn(
          "agpt-border-input agpt-shadow-input flex h-9 w-full rounded-md border bg-transparent px-3 py-1 text-body leading-5 transition-colors file:border-0 file:bg-transparent file:text-body file:font-medium file:leading-5 placeholder:text-placeholder disabled:cursor-not-allowed disabled:opacity-50",
          type == "file" ? "pb-0.5 pt-1.5" : "",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
