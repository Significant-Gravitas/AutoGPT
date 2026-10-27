import { cn } from "@/lib/utils";
import { forwardRef } from "react";

export type BaseInputProps = React.InputHTMLAttributes<HTMLInputElement>;

// The bare <input> under the Input atom, for fields that bring their own label
// and layout.
// TODO(tokens): no role is gray-500 #6B7280 (`placeholder` is #83838C). The
// dark: class never applies (forcedTheme="light") and is DS6's (SECRT-3116) to
// delete.
export const BaseInput = forwardRef<HTMLInputElement, BaseInputProps>(
  function BaseInput({ className, type, ...props }, ref) {
    return (
      <input
        type={type}
        className={cn(
          "agpt-border-input agpt-shadow-input flex h-9 w-full rounded-md border bg-transparent px-3 py-1 text-body leading-5 transition-colors file:border-0 file:bg-transparent file:text-body file:font-medium file:leading-5 placeholder:text-gray-500 disabled:cursor-not-allowed disabled:opacity-50 dark:placeholder:text-gray-400",
          type == "file" ? "pb-0.5 pt-1.5" : "",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
